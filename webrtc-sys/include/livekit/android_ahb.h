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

#pragma once

#include <jni.h>

namespace livekit_ffi {

/// Called when the encoder has finished with a hardware buffer frame. `fence_fd` is a native
/// fence behind the encoder's GPU reads, or -1; the handler owns the fd. Called on the
/// helper's GL thread.
typedef void (*AhbReleaseHandler)(void* user, void* hardware_buffer, int fence_fd);

/// The EGL context the hardware encoder should share, so texture frames can be drawn into
/// its input surface. Null (a local reference otherwise) when livekit-ahb.jar is not in the
/// app, in which case the encoder keeps its byte-buffer path.
jobject GetAhbSharedEglContext(JNIEnv* env);

void SetAhbReleaseHandler(AhbReleaseHandler handler, void* user);

}  // namespace livekit_ffi
