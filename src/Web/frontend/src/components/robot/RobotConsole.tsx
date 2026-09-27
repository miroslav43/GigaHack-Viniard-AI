"use client";

// The robot console: live camera, photos, the camera (motor 2 turns it, motor 1 raises it) and the wheels, hold to
// move; the record mode (panoramas every few tens of cm); the settings.
// Keyboard: hold the arrows (camera) or W A S D (wheels); X / Esc stops the wheels; space takes a photo.
import { useCallback, useEffect, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CenterFocusStrongOutlined from "@mui/icons-material/CenterFocusStrongOutlined";
import FlashlightOnOutlined from "@mui/icons-material/FlashlightOnOutlined";
import PhotoCameraOutlined from "@mui/icons-material/PhotoCameraOutlined";
import StopCircleOutlined from "@mui/icons-material/StopCircleOutlined";
import VideocamOffOutlined from "@mui/icons-material/VideocamOffOutlined";
import { useTranslations } from "next-intl";
import StraightenOutlined from "@mui/icons-material/StraightenOutlined";
import { useFormat } from "@/lib/useFormat";
import { parseBoardUrl } from "@/lib/robot/commands";
import { ArrowPad, type Arrow } from "./ArrowPad";
import { PhotoGallery, type Photo } from "./PhotoGallery";
import { PanoramaGallery } from "./PanoramaGallery";
import { RecordCard } from "./RecordCard";
import { SettingsCard } from "./SettingsCard";
import { callRobot, RobotCallError, type RobotError } from "./robotApi";
import { useRobotSettings } from "./useRobotSettings";
import { useCameraFrames } from "./useCameraFrames";
import { useRobotMotion } from "./useRobotMotion";
import { useRecorder, type Panorama } from "./useRecorder";
import { orientCss, orientJpeg } from "./orient";

/** how often the distance sensor is read (ms): every second, and 4 times a second while the robot goes forward */
const DISTANCE_EVERY_MS = 1000;
const DISTANCE_FORWARD_MS = 250;
const KEYS: Record<string, { kind: "camera" | "drive"; arrow: Arrow }> = {
  ArrowUp: { kind: "camera", arrow: "up" },
  ArrowDown: { kind: "camera", arrow: "down" },
  ArrowLeft: { kind: "camera", arrow: "left" },
  ArrowRight: { kind: "camera", arrow: "right" },
  w: { kind: "drive", arrow: "up" },
  s: { kind: "drive", arrow: "down" },
  a: { kind: "drive", arrow: "left" },
  d: { kind: "drive", arrow: "right" },
};

type Status = { severity: "success" | "info" | "error"; text: string } | null;

/** The sensor's answer ("213.5 cm", or its out-of-range text) as centimetres, or null. */
const centimetres = (text: string | null) => {
  const m = text ? /^(-?\d+(?:\.\d+)?)\s*cm$/.exec(text.trim()) : null;
  return m ? Number(m[1]) : null;
};

/** Object URLs of a list, released when the page goes away. */
function useReleaseOnUnmount<T>(items: T[], urls: (item: T) => string[]) {
  const ref = useRef(items);
  useEffect(() => {
    ref.current = items;
  }, [items]);
  const urlsRef = useRef(urls);
  useEffect(() => () => ref.current.flatMap((i) => urlsRef.current(i)).forEach((u) => URL.revokeObjectURL(u)), []);
}

export function RobotConsole() {
  const t = useTranslations("robot");
  const f = useFormat();
  const { settings, update } = useRobotSettings();
  const cam = parseBoardUrl(settings.urls.cam);
  const motors = parseBoardUrl(settings.urls.motors);
  const drive = parseBoardUrl(settings.urls.drive);

  const [streamOn, setStreamOn] = useState(true);
  const frames = useCameraFrames(cam, streamOn);
  const [status, setStatus] = useState<Status>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [panoramas, setPanoramas] = useState<Panorama[]>([]);
  const [distance, setDistance] = useState<string | null>(null);
  useReleaseOnUnmount(photos, (p) => [p.url]);
  useReleaseOnUnmount(panoramas, (p) => [p.strip.url, ...p.frames.map((f) => f.url)]);

  const fail = useCallback((e: unknown) => {
    const code: RobotError = e instanceof RobotCallError ? e.code : "failed";
    setStatus({ severity: "error", text: t(`errors.${code}`) });
  }, [t]);

  const motion = useRobotMotion({ motors, drive, settings, onError: fail });
  const recorder = useRecorder({
    cam,
    settings,
    panTo: motion.panTo,
    forwardFor: motion.forwardFor,
    stopWheels: motion.stopWheels,
    onPanorama: useCallback((p: Panorama) => setPanoramas((ps) => [p, ...ps]), []),
    onError: fail,
  });

  // obstacle guard: something closer than `safeStopCm` in front of the sensor blocks going forward (0 = off)
  const cm = centimetres(distance);
  const blocked = settings.safeStopCm > 0 && cm !== null && cm < settings.safeStopCm;
  const goingForward = (motion.holding?.kind === "drive" && motion.holding.arrow === "up") || recorder.phase?.step === "drive";

  const hold = useCallback(
    (kind: "camera" | "drive", a: Arrow) => {
      if (recorder.recording) return;
      if (kind === "drive" && a === "up" && blocked) {
        setStatus({ severity: "error", text: t("status.obstacle", { cm: f.num(cm!, 1), limit: settings.safeStopCm }) });
        return;
      }
      setStatus({ severity: "info", text: kind === "camera" ? t("status.cameraMoving", { dir: t(`camera.${a}`) }) : t("status.driving", { dir: t(`drive.${({ up: "forward", down: "back", left: "left", right: "right" } as const)[a]}`) }) });
      if (kind === "camera") void motion.holdCamera(a);
      else motion.holdDrive(a);
    },
    [recorder.recording, blocked, cm, settings.safeStopCm, motion, t, f],
  );
  const release = useCallback(() => {
    // clears a "moving…" note; an error (e.g. the obstacle that stopped the wheels) stays
    void motion.release().then(() => setStatus((s) => (s?.severity === "info" ? null : s)));
  }, [motion]);
  const stopAll = useCallback(async () => {
    await motion.release();
    await motion.stopWheels();
    setStatus({ severity: "success", text: t("status.stopped") });
  }, [motion, t]);

  // the distance sensor (HC-SR04 on the camera's motor board); not while the camera moves (the board answers one
  // request at a time)
  const cameraMoving = motion.holding?.kind === "camera" || recorder.phase?.step === "photo" || recorder.phase?.step === "back";
  useEffect(() => {
    if (!motors || cameraMoving) return;
    let alive = true;
    const read = () =>
      callRobot("motors", motors, { cmd: "distance" })
        .then((r) => r.text())
        .then((text) => alive && setDistance(text.trim()))
        .catch(() => alive && setDistance(null));
    const timer = setInterval(read, goingForward ? DISTANCE_FORWARD_MS : DISTANCE_EVERY_MS);
    void read();
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [motors, cameraMoving, goingForward]);

  // an obstacle while going forward: the wheels stop at once (a held button, or the panorama's drive)
  const { release: releaseHold, stopWheels } = motion;
  const stopRecorder = recorder.stop;
  useEffect(() => {
    if (!blocked || !goingForward) return;
    const text = t("status.obstacle", { cm: f.num(cm!, 1), limit: settings.safeStopCm });
    void (recorder.recording ? stopRecorder() : releaseHold().then(stopWheels)).then(() => setStatus({ severity: "error", text }));
  }, [blocked, goingForward, recorder.recording, stopRecorder, releaseHold, stopWheels, cm, settings.safeStopCm, t, f]);

  const [capturing, setCapturing] = useState(false);
  const takePhoto = useCallback(async () => {
    if (!cam || capturing) return;
    setCapturing(true);
    setStatus({ severity: "info", text: t("status.capturing") });
    try {
      // the frame on screen when the live picture runs, otherwise a fresh one from the camera
      const raw = streamOn && frames.blob && !frames.down ? frames.blob : await (await callRobot("cam", cam, { cmd: "capture" })).blob();
      if (!raw.type.startsWith("image/")) throw new RobotCallError("failed");
      const blob = await orientJpeg(raw, settings); // saved the way it is shown
      const pose = motion.poseRef.current;
      setPhotos((ps) => [{ id: crypto.randomUUID(), url: URL.createObjectURL(blob), takenAt: new Date(), panDeg: pose.pan, heightSteps: pose.height }, ...ps]);
      setStatus({ severity: "success", text: t("status.captured") });
    } catch (e) {
      fail(e);
    } finally {
      setCapturing(false);
    }
  }, [cam, capturing, streamOn, frames, motion.poseRef, settings, t, fail]);

  // between two frames the camera is free: the flash goes through while the live picture runs
  const flash = useCallback(async () => {
    if (!cam) return;
    try {
      const res = await callRobot("cam", cam, { cmd: "flash_toggle" });
      setStatus({ severity: "success", text: t("status.flash", { state: (await res.text()).trim() }) });
    } catch (e) {
      fail(e);
    }
  }, [cam, t, fail]);

  // keyboard: hold an arrow / W A S D, except while typing in a field
  useEffect(() => {
    const typing = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      return Boolean(el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable));
    };
    const onDown = (e: KeyboardEvent) => {
      if (typing(e)) return;
      if (e.key === " ") {
        e.preventDefault();
        if (!e.repeat) void takePhoto();
        return;
      }
      if (e.key === "Escape" || e.key.toLowerCase() === "x") {
        void (recorder.recording ? recorder.stop() : stopAll());
        return;
      }
      const k = KEYS[e.key] ?? KEYS[e.key.toLowerCase()];
      if (!k) return;
      e.preventDefault();
      if (!e.repeat) hold(k.kind, k.arrow);
    };
    const onUp = (e: KeyboardEvent) => {
      if (KEYS[e.key] ?? KEYS[e.key.toLowerCase()]) release();
    };
    // leaving the tab ends a hold as well (its key-up would never come)
    const onBlur = () => release();
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [takePhoto, hold, release, stopAll, recorder]);

  const cameraLabels = { up: t("camera.up"), down: t("camera.down"), left: t("camera.left"), right: t("camera.right") };
  const driveLabels = { up: t("drive.forward"), down: t("drive.back"), left: t("drive.left"), right: t("drive.right") };
  const showStream = cam && streamOn && frames.url && !frames.down;
  const held = (kind: "camera" | "drive") => (motion.holding?.kind === kind ? motion.holding.arrow : null);

  return (
    <Box sx={{ display: "grid", gap: 4 }}>
      <Box sx={{ display: "grid", gap: 4, gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 3fr) minmax(320px, 2fr)" }, alignItems: "start" }}>
        {/* ---- live camera + record mode ---- */}
        <Box sx={{ display: "grid", gap: 4 }}>
          <Paper sx={{ p: 4 }}>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, gap: 2, flexWrap: "wrap" }}>
              <Typography variant="h3">{t("live.title")}</Typography>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Tooltip title={t("live.flash")}>
                  <span>
                    <IconButton onClick={flash} disabled={!cam} aria-label={t("live.flash")}>
                      <FlashlightOnOutlined />
                    </IconButton>
                  </span>
                </Tooltip>
                <Button size="small" onClick={() => setStreamOn((v) => !v)} disabled={!cam}>
                  {streamOn ? t("live.stop") : t("live.start")}
                </Button>
                <Button variant="contained" startIcon={<PhotoCameraOutlined />} onClick={takePhoto} disabled={!cam || capturing} data-testid="robot-capture">
                  {t("live.capture")}
                </Button>
              </Box>
            </Box>
            <Box sx={{ position: "relative", aspectRatio: "4 / 3", bgcolor: "grey.900", borderRadius: 2, overflow: "hidden", display: "grid", placeItems: "center" }}>
              {showStream ? (
                <>
                  {/* the latest frame (object URL of a /capture); a new one replaces it as soon as it arrives */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={frames.url!} alt={t("live.title")} data-testid="robot-frame" style={{ width: "100%", height: "100%", objectFit: "contain", transform: orientCss(settings) }} />
                  <Typography variant="caption" sx={{ position: "absolute", right: 8, bottom: 6, color: "common.white", textShadow: "0 0 3px black" }}>
                    {t("live.fps", { fps: frames.fps.toFixed(1) })}
                  </Typography>
                  {/* the distance sensor (HC-SR04), in the corner of the live picture */}
                  {motors && (
                    <Box
                      data-testid="robot-live-distance"
                      data-blocked={blocked || undefined}
                      sx={{
                        position: "absolute", left: 8, top: 8, display: "flex", alignItems: "center", gap: 0.75,
                        px: 1.25, py: 0.5, borderRadius: 1.5, boxShadow: 2,
                        bgcolor: blocked ? "error.main" : "background.paper", color: blocked ? "error.contrastText" : "text.primary",
                      }}
                    >
                      <StraightenOutlined fontSize="small" color={blocked ? "inherit" : "primary"} />
                      <Typography variant="subtitle2" sx={{ fontVariantNumeric: "tabular-nums" }}>
                        {cm !== null ? `${f.num(cm, 1)} cm` : distance ? t("live.outOfRange") : "—"}
                        {blocked && ` · ${t("live.obstacle")}`}
                      </Typography>
                    </Box>
                  )}
                </>
              ) : (
                <Box sx={{ color: "grey.400", textAlign: "center", px: 4 }}>
                  <VideocamOffOutlined fontSize="large" />
                  <Typography variant="body2">
                    {!cam ? t("live.noAddress") : !streamOn ? t("live.off") : frames.down ? t("live.failed", { url: cam }) : t("live.connecting")}
                  </Typography>
                </Box>
              )}
            </Box>
            {status && (
              <Alert severity={status.severity} sx={{ mt: 2 }} data-testid="robot-status">
                {status.text}
              </Alert>
            )}
          </Paper>
          <RecordCard settings={settings} ready={Boolean(cam && motors && drive) && !blocked} phase={recorder.phase} onStart={() => void recorder.start()} onStop={() => void recorder.stop()} />
        </Box>

        {/* ---- controls ---- */}
        <Box sx={{ display: "grid", gap: 4 }}>
          <Paper sx={{ p: 4 }}>
            <Typography variant="h3" sx={{ mb: 1 }}>
              {t("camera.title")}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {t("camera.hint", { pan: Math.round(motion.pose.pan), height: motion.pose.height })}
            </Typography>
            {motors && (
              <Typography variant="subtitle2" sx={{ mb: 2 }} data-testid="robot-distance">
                {t("camera.distance", { value: distance ?? "—" })}
              </Typography>
            )}
            <ArrowPad
              testId="robot-camera-pad"
              labels={cameraLabels}
              disabled={!motors || recorder.recording}
              active={held("camera")}
              onHold={(a) => hold("camera", a)}
              onRelease={release}
              centre={
                <Tooltip title={t("camera.zero")}>
                  <IconButton onClick={motion.zero} aria-label={t("camera.zero")}>
                    <CenterFocusStrongOutlined />
                  </IconButton>
                </Tooltip>
              }
            />
            {!motors && (
              <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 2, textAlign: "center" }}>
                {t("camera.noAddress")}
              </Typography>
            )}
          </Paper>
          <Paper sx={{ p: 4 }}>
            <Typography variant="h3" sx={{ mb: 1 }}>
              {t("drive.title")}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {t("drive.hint", { speed: settings.wheelSpeed })}
            </Typography>
            <ArrowPad
              testId="robot-drive-pad"
              labels={driveLabels}
              disabled={!drive || recorder.recording}
              disabledArrows={blocked ? ["up"] : []}
              active={held("drive")}
              onHold={(a) => hold("drive", a)}
              onRelease={release}
              centre={
                <Tooltip title={t("drive.stop")}>
                  <span>
                    <IconButton color="error" onClick={() => void stopAll()} disabled={!drive} aria-label={t("drive.stop")}>
                      <StopCircleOutlined fontSize="large" />
                    </IconButton>
                  </span>
                </Tooltip>
              }
            />
            {!drive && (
              <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 2, textAlign: "center" }}>
                {t("drive.noAddress")}
              </Typography>
            )}
          </Paper>
          <SettingsCard settings={settings} onChange={update} />
        </Box>
      </Box>
      <PanoramaGallery
        panoramas={panoramas}
        onRemove={(id) =>
          setPanoramas((ps) => {
            const p = ps.find((x) => x.id === id);
            if (p) [p.strip.url, ...p.frames.map((f) => f.url)].forEach((u) => URL.revokeObjectURL(u));
            return ps.filter((x) => x.id !== id);
          })
        }
      />
      <PhotoGallery
        photos={photos}
        onRemove={(id) =>
          setPhotos((ps) => {
            const p = ps.find((x) => x.id === id);
            if (p) URL.revokeObjectURL(p.url);
            return ps.filter((x) => x.id !== id);
          })
        }
      />
      <Typography variant="caption" color="text.secondary">
        {t("keys")}
      </Typography>
    </Box>
  );
}
