# TOP-5 additional features · Solemtrix Hardware & Software

These five features are outside the challenge requirements: they are not canopies, waste, row axes, attributes,
measurements or the route. Each one is built and shown on slide 5 of `gigahack/gigahack.pdf`.

1. **Field robot.** A working ESP32 ground robot driven from the web app. It has a live camera with pan and tilt, one-press panoramas, hold-to-drive and an obstacle stop at 20 cm.
2. **Grape and disease detector.** Our own YOLO detector (YOLO11 / YOLO26), fine-tuned on public vineyard datasets. It detects grape bunches (WGISD, ViViD-5K), trunks, and diseased leaves (Flavescence dorée). The code is in `model_detectie_vita_de_vie/`.
3. **Cadastre and farm registry.** Every vineyard block is linked to its official AGCC cadastre parcels, with their codes and land use. Blocks are grouped into 27 farms, and public and internal roads are separate layers. This serves the vine registry and AIPA subsidy checks.
4. **Inspector team and tasks.** Each municipality has its own accounts: an admin, inspectors and viewers. The admin assigns one spot or a whole farm to an inspector, who gets a notification, and finished tasks are tracked. Each municipality sees only its own data.
5. **In-app AI assistant.** It answers "how do I…?" questions for each role, in RO, EN and RU, and quotes the exact button labels shown on screen. It runs on Gemini, and the API keys stay on the server.
