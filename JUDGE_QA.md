# Judge Q&A

## 1. How is KINESYS different from AirTouch or Neural Lab?

KINESYS is not just gesture recognition. It is a context-aware gesture OS layer. The same gesture maps differently per active application, supports two-hand modifiers, adapts to fatigue, and allows personal user-specific training.

## 2. What is your gesture recognition accuracy?

The exact number depends on lighting, framing, and the user, but the runtime reduces false triggers with a 3-frame hold requirement, a 0.75 confidence threshold, and optional personal KNN training for user-specific adaptation.

## 3. Why KNN and not a neural network for personal training?

KNN is the right hackathon choice for 5-shot personalization. It trains instantly, works well on small landmark vectors, is easy to update live, and avoids the latency and data requirements of retraining a neural network per user.

## 4. What happens when lighting conditions are poor?

MediaPipe quality drops first, so KINESYS falls back to safer behavior: fewer actions fire because of the hold and confidence gates. The user can also lock input instantly with a closed fist and use the backup prerecorded demo if needed.

## 5. How long does personal gesture training take?

Roughly a few seconds per gesture. The user records 5 samples, the KNN model is fit immediately, and the updated gesture becomes available without restarting the app.

## 6. Can this work on Mac or Linux?

The current hackathon build is Windows-first because app detection and OS automation rely on `pywin32`, `win32gui`, and Windows input behavior. The gesture layer itself is portable, but the system integration would need platform-specific adapters.

## 7. What is the end-to-end latency?

The design target is around webcam frame time, about 33 ms per frame at 30 FPS, plus gesture hold gating for intentional actions. In practice, cursor movement feels near real time, while discrete actions wait for three stable frames for safety.

## 8. How do you prevent false gesture triggers?

KINESYS combines multiple safeguards: 3 consecutive hold frames, confidence thresholding, action debounce, lock state, fatigue-adaptive smoothing, and a rare two-hand X gesture for termination.

## 9. What if the active app is not in your profiles?

The system falls back automatically to `profiles/default.json`, so gesture handling remains stable even for unknown or unsupported applications.

## 10. How would you monetize KINESYS?

The clearest paths are accessibility software, enterprise productivity integrations, industrial hands-free control, and a developer API for app-specific gesture plugins.
