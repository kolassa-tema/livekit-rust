/*
 * Copyright 2025 LiveKit, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Zero-copy video frames on Android: an AHardwareBuffer produced by the app's GPU is wrapped
// as a GL texture on an EGL context shared with the hardware encoder, so
// HardwareVideoEncoder draws it straight into the MediaCodec input surface. No CPU copy, no
// colour conversion. The GL side lives in the Java helper livekit.tema.AhbTextureSource
// (shipped as livekit-ahb.jar next to libwebrtc.jar); the EGL extension calls it needs are
// implemented here and registered as its natives, so the jar has no JNI exports to maintain.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <unistd.h>

#include <mutex>

#include "livekit/android_ahb.h"
#include "livekit/video_frame_buffer.h"
#include "rtc_base/logging.h"
#include "sdk/android/native_api/jni/class_loader.h"
#include "sdk/android/native_api/jni/scoped_java_ref.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/video_frame.h"

namespace livekit_ffi {
namespace {

#define AHB_LOG(...) __android_log_print(ANDROID_LOG_INFO, "livekit_ahb", __VA_ARGS__)

const char kHelperClass[] = "livekit/tema/AhbTextureSource";

std::mutex g_mutex;
bool g_helper_failed = false;
jobject g_helper = nullptr;  // global ref to the singleton
jmethodID g_wrap = nullptr;
jmethodID g_shared_context = nullptr;
jmethodID g_buffer_release = nullptr;

AhbReleaseHandler g_release_handler = nullptr;
void* g_release_user = nullptr;

// EGL and GLES extension entry points, resolved once on first use.
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC p_eglGetNativeClientBufferANDROID = nullptr;
PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR = nullptr;
PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR = nullptr;
PFNEGLCREATESYNCKHRPROC p_eglCreateSyncKHR = nullptr;
PFNEGLDESTROYSYNCKHRPROC p_eglDestroySyncKHR = nullptr;
PFNEGLWAITSYNCKHRPROC p_eglWaitSyncKHR = nullptr;
PFNEGLDUPNATIVEFENCEFDANDROIDPROC p_eglDupNativeFenceFDANDROID = nullptr;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES = nullptr;

// The FFI is linked for an older platform level than these API 26 symbols, which makes them
// weak and null at runtime; resolve them from libandroid.so instead of trusting the link.
typedef void (*PFN_AHardwareBuffer_acquire)(AHardwareBuffer*);
typedef void (*PFN_AHardwareBuffer_release)(AHardwareBuffer*);
PFN_AHardwareBuffer_acquire p_AHardwareBuffer_acquire = nullptr;
PFN_AHardwareBuffer_release p_AHardwareBuffer_release = nullptr;

bool ResolveExtensions() {
  static bool resolved = false;
  static bool ok = false;
  if (resolved) return ok;
  resolved = true;
#define LK_RESOLVE(type, name) \
  p_##name = reinterpret_cast<type>(eglGetProcAddress(#name)); \
  if (!p_##name) { RTC_LOG(LS_ERROR) << "[ahb] missing " #name; return false; }
  LK_RESOLVE(PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC, eglGetNativeClientBufferANDROID);
  LK_RESOLVE(PFNEGLCREATEIMAGEKHRPROC, eglCreateImageKHR);
  LK_RESOLVE(PFNEGLDESTROYIMAGEKHRPROC, eglDestroyImageKHR);
  LK_RESOLVE(PFNEGLCREATESYNCKHRPROC, eglCreateSyncKHR);
  LK_RESOLVE(PFNEGLDESTROYSYNCKHRPROC, eglDestroySyncKHR);
  LK_RESOLVE(PFNEGLWAITSYNCKHRPROC, eglWaitSyncKHR);
  LK_RESOLVE(PFNEGLDUPNATIVEFENCEFDANDROIDPROC, eglDupNativeFenceFDANDROID);
  LK_RESOLVE(PFNGLEGLIMAGETARGETTEXTURE2DOESPROC, glEGLImageTargetTexture2DOES);
#undef LK_RESOLVE
  void* libandroid = dlopen("libandroid.so", RTLD_NOW | RTLD_NOLOAD);
  if (!libandroid) libandroid = dlopen("libandroid.so", RTLD_NOW);
  if (libandroid) {
    p_AHardwareBuffer_acquire =
        reinterpret_cast<PFN_AHardwareBuffer_acquire>(dlsym(libandroid, "AHardwareBuffer_acquire"));
    p_AHardwareBuffer_release =
        reinterpret_cast<PFN_AHardwareBuffer_release>(dlsym(libandroid, "AHardwareBuffer_release"));
  }
  if (!p_AHardwareBuffer_acquire || !p_AHardwareBuffer_release) {
    AHB_LOG("AHardwareBuffer_acquire/release not found in libandroid.so");
    return false;
  }
  ok = true;
  AHB_LOG("EGL, GLES and AHardwareBuffer entry points resolved");
  return true;
}

// ---- natives of livekit.tema.AhbTextureSource, all called on its GL thread ----

// Imports the buffer into the current EGL context as `texture`, waiting on `sync_fd` first
// if one was given. Returns the EGLImage, or 0 on failure. Takes a reference on the buffer.
jlong NativeCreateImage(JNIEnv*, jclass, jlong ahb, jint sync_fd, jint texture) {
  if (!ResolveExtensions()) return 0;
  EGLDisplay display = eglGetCurrentDisplay();
  auto* buffer = reinterpret_cast<AHardwareBuffer*>(ahb);
  p_AHardwareBuffer_acquire(buffer);

  EGLClientBuffer client = p_eglGetNativeClientBufferANDROID(buffer);
  const EGLint attrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
  EGLImageKHR image = p_eglCreateImageKHR(display, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                                          client, attrs);
  if (image == EGL_NO_IMAGE_KHR) {
    RTC_LOG(LS_ERROR) << "[ahb] eglCreateImageKHR failed, egl error " << static_cast<int>(eglGetError());
    p_AHardwareBuffer_release(buffer);
    if (sync_fd >= 0) close(sync_fd);
    return 0;
  }

  if (sync_fd >= 0) {
    // The producer's GPU work has to land before the encoder samples; this makes the GL
    // queue wait without stalling this thread. EGL owns the fd once the sync exists.
    const EGLint sync_attrs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, sync_fd, EGL_NONE};
    EGLSyncKHR sync = p_eglCreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, sync_attrs);
    if (sync != EGL_NO_SYNC_KHR) {
      p_eglWaitSyncKHR(display, sync, 0);
      p_eglDestroySyncKHR(display, sync);
    } else {
      RTC_LOG(LS_WARNING) << "[ahb] eglCreateSyncKHR from fd failed; proceeding unsynchronised";
      close(sync_fd);
    }
  }

  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
  p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  return reinterpret_cast<jlong>(image);
}

void NativeDestroyImage(JNIEnv*, jclass, jlong ahb, jlong image) {
  if (image && p_eglDestroyImageKHR)
    p_eglDestroyImageKHR(eglGetCurrentDisplay(), reinterpret_cast<EGLImageKHR>(image));
  if (p_AHardwareBuffer_release) p_AHardwareBuffer_release(reinterpret_cast<AHardwareBuffer*>(ahb));
}

// A fence behind everything the GL queue has been given so far, as a sync fd the producer
// can wait on (Vulkan imports it as a semaphore) before it writes the buffer again.
jint NativeCreateFence(JNIEnv*, jclass) {
  if (!ResolveExtensions()) return -1;
  EGLDisplay display = eglGetCurrentDisplay();
  EGLSyncKHR sync = p_eglCreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
  if (sync == EGL_NO_SYNC_KHR) return -1;
  glFlush();
  int fd = p_eglDupNativeFenceFDANDROID(display, sync);
  p_eglDestroySyncKHR(display, sync);
  return fd;
}

void NativeOnReleased(JNIEnv*, jclass, jlong ahb, jint fence_fd) {
  AhbReleaseHandler handler;
  void* user;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    handler = g_release_handler;
    user = g_release_user;
  }
  if (handler) {
    handler(user, reinterpret_cast<void*>(ahb), fence_fd);
  } else if (fence_fd >= 0) {
    close(fence_fd);
  }
}

const JNINativeMethod kNatives[] = {
    {"nativeCreateImage", "(JII)J", reinterpret_cast<void*>(NativeCreateImage)},
    {"nativeDestroyImage", "(JJ)V", reinterpret_cast<void*>(NativeDestroyImage)},
    {"nativeCreateFence", "()I", reinterpret_cast<void*>(NativeCreateFence)},
    {"nativeOnReleased", "(JI)V", reinterpret_cast<void*>(NativeOnReleased)},
};

// Finds the helper, registers its natives and creates its singleton. False when the jar is
// not in the app; callers then behave exactly as before this file existed.
bool EnsureHelper(JNIEnv* env) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_helper) return true;
  if (g_helper_failed) return false;

  AHB_LOG("EnsureHelper: looking up %s", kHelperClass);
  webrtc::ScopedJavaLocalRef<jclass> cls = webrtc::GetClass(env, kHelperClass);
  if (cls.is_null() || env->ExceptionCheck()) {
    env->ExceptionClear();
    RTC_LOG(LS_WARNING) << "[ahb] " << kHelperClass
                        << " not found; hardware buffer frames and encoder surface mode disabled";
    g_helper_failed = true;
    return false;
  }

  if (env->RegisterNatives(cls.obj(), kNatives, sizeof(kNatives) / sizeof(kNatives[0])) != JNI_OK) {
    env->ExceptionClear();
    RTC_LOG(LS_ERROR) << "[ahb] RegisterNatives failed";
    g_helper_failed = true;
    return false;
  }

  jmethodID get = env->GetStaticMethodID(cls.obj(), "get", "()Llivekit/tema/AhbTextureSource;");
  g_wrap = env->GetMethodID(cls.obj(), "wrap", "(JIII)Llivekit/org/webrtc/VideoFrame$Buffer;");
  g_shared_context = env->GetMethodID(cls.obj(), "getSharedContext",
                                      "()Llivekit/org/webrtc/EglBase$Context;");
  webrtc::ScopedJavaLocalRef<jclass> buffer_cls =
      webrtc::GetClass(env, "livekit/org/webrtc/VideoFrame$Buffer");
  g_buffer_release = env->GetMethodID(buffer_cls.obj(), "release", "()V");
  if (!get || !g_wrap || !g_shared_context || !g_buffer_release || env->ExceptionCheck()) {
    env->ExceptionClear();
    RTC_LOG(LS_ERROR) << "[ahb] helper methods not found; is livekit-ahb.jar older than livekit_ffi?";
    g_helper_failed = true;
    return false;
  }

  AHB_LOG("EnsureHelper: natives registered, calling get()");
  jobject helper = env->CallStaticObjectMethod(cls.obj(), get);
  AHB_LOG("EnsureHelper: get() returned %p", helper);
  if (!helper || env->ExceptionCheck()) {
    env->ExceptionDescribe();
    env->ExceptionClear();
    RTC_LOG(LS_ERROR) << "[ahb] AhbTextureSource.get() failed (no EGL?)";
    g_helper_failed = true;
    return false;
  }
  g_helper = env->NewGlobalRef(helper);
  env->DeleteLocalRef(helper);
  RTC_LOG(LS_INFO) << "[ahb] AhbTextureSource ready; encoder gets a shared EGL context";
  AHB_LOG("AhbTextureSource ready");
  return true;
}

}  // namespace

jobject GetAhbSharedEglContext(JNIEnv* env) {
  if (!EnsureHelper(env)) return nullptr;
  jobject context = env->CallObjectMethod(g_helper, g_shared_context);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return nullptr;
  }
  return context;
}

void SetAhbReleaseHandler(AhbReleaseHandler handler, void* user) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_release_handler = handler;
  g_release_user = user;
}

std::unique_ptr<VideoFrameBuffer> new_native_buffer_from_hardware_buffer(size_t hardware_buffer,
                                                                        int32_t sync_fd,
                                                                        uint32_t width,
                                                                        uint32_t height) {
  JNIEnv* env = webrtc::AttachCurrentThreadIfNeeded();
  if (!hardware_buffer || !EnsureHelper(env)) {
    if (sync_fd >= 0) close(sync_fd);
    return nullptr;
  }

  jobject j_buffer = env->CallObjectMethod(g_helper, g_wrap, static_cast<jlong>(hardware_buffer),
                                           static_cast<jint>(sync_fd), static_cast<jint>(width),
                                           static_cast<jint>(height));
  if (env->ExceptionCheck() || !j_buffer) {
    env->ExceptionDescribe();
    env->ExceptionClear();
    RTC_LOG(LS_ERROR) << "[ahb] wrap failed for buffer " << hardware_buffer;
    return nullptr;
  }

  // JavaToNativeFrameBuffer retains the Java buffer for the native wrapper's lifetime; the
  // reference wrap() handed us is dropped here so the texture dies with the native frame.
  // Adopts the local reference wrap() returned; it is released with j_ref.
  webrtc::ScopedJavaLocalRef<jobject> j_ref = webrtc::ScopedJavaLocalRef<jobject>::Adopt(env, j_buffer);
  webrtc::scoped_refptr<webrtc::VideoFrameBuffer> native =
      webrtc::jni::JavaToNativeFrameBuffer(env, j_ref);
  env->CallVoidMethod(j_ref.obj(), g_buffer_release);
  if (!native) return nullptr;
  return std::make_unique<VideoFrameBuffer>(native);
}

}  // namespace livekit_ffi
