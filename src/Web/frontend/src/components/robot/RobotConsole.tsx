"use client";

// The robot console: live camera, photos, the camera's pan / tilt (steppers 1 and 2), the wheels, and the settings.
// Keyboard: arrows move the camera, W A S D drive, space takes a photo (not while typing in a field).
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
import { parseBoardUrl } from "@/lib/robot/commands";
import { wheelDirs, type DriveMove } from "@/lib/robot/wheels";
import { ArrowPad, type Arrow } from "./ArrowPad";
import { PhotoGallery, type Photo } from "./PhotoGallery";
import { SettingsCard } from "./SettingsCard";
import { callRobot, RobotCallError, type RobotError } from "./robotApi";
import { useRobotSettings } from "./useRobotSettings";
import { useCameraFrames } from "./useCameraFrames";

const DRIVE_OF: Record<Arrow, DriveMove> = { up: "forward", down: "back", left: "left", right: "right" };
/** how often the distance sensor is read (ms) */
const DISTANCE_EVERY_MS = 1000;
/** camera arrows → stepper: motor 2 turns the camera (left / right), motor 1 raises and lowers it (checked on the
 *  robot, 27.09.2026); dir 0 = right / up unless the axis is set as reversed */
const CAMERA_OF: Record<Arrow, { motor: 1 | 2; axis: "pan" | "height"; sign: 1 | -1 }> = {
  right: { motor: 2, axis: "pan", sign: 1 },
  left: { motor: 2, axis: "pan", sign: -1 },
  up: { motor: 1, axis: "height", sign: 1 },
  down: { motor: 1, axis: "height", sign: -1 },
};
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

export function RobotConsole() {
  const t = useTranslations("robot");
  const { settings, update } = useRobotSettings();
  const cam = parseBoardUrl(settings.urls.cam);
  const motors = parseBoardUrl(settings.urls.motors);
  const drive = parseBoardUrl(settings.urls.drive);

  const [streamOn, setStreamOn] = useState(true);
  const frames = useCameraFrames(cam, streamOn);
  const [busy, setBusy] = useState<"camera" | "drive" | "photo" | null>(null);
  const [status, setStatus] = useState<Status>(null);
  // estimated camera position: pan in degrees, height in steps, from the zeroed position
  const [pose, setPose] = useState({ pan: 0, height: 0 });
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [distance, setDistance] = useState<string | null>(null);
  // the object URLs of the photos are released when the page goes away
  const photosRef = useRef(photos);
  useEffect(() => {
    photosRef.current = photos;
  }, [photos]);
  useEffect(() => () => photosRef.current.forEach((p) => URL.revokeObjectURL(p.url)), []);

  const fail = useCallback((e: unknown) => {
    const code: RobotError = e instanceof RobotCallError ? e.code : "failed";
    setStatus({ severity: "error", text: t(`errors.${code}`) });
  }, [t]);

  const moveCamera = useCallback(
    async (a: Arrow) => {
      if (!motors || busy) return;
      const m = CAMERA_OF[a];
      const pan = m.axis === "pan";
      const steps = pan ? Math.max(1, Math.round((settings.stepDeg / 360) * settings.stepsPerRev)) : settings.liftSteps;
      const reversed = pan ? settings.panInvert : settings.liftInvert;
      const dir = (m.sign > 0) !== reversed ? 0 : 1;
      setBusy("camera");
      setStatus({ severity: "info", text: t("status.cameraMoving", { dir: t(`camera.${a}`) }) });
      try {
        await callRobot("motors", motors, { cmd: "move", motor: m.motor, dir, speed: pan ? settings.speedPps : settings.liftSpeedPps, steps });
        const change = pan ? (steps * 360) / settings.stepsPerRev : steps;
        setPose((s) => ({ ...s, [m.axis]: s[m.axis] + m.sign * change }));
        setStatus({ severity: "success", text: t("status.cameraDone") });
      } catch (e) {
        fail(e);
      } finally {
        setBusy(null);
      }
    },
    [motors, busy, settings, t, fail],
  );

  const driveTo = useCallback(
    async (move: DriveMove) => {
      if (!drive || busy) return;
      setBusy("drive");
      setStatus({ severity: "info", text: t("status.driving", { dir: t(`drive.${move}`) }) });
      try {
        const dirs = wheelDirs(move, settings.wheelSide, settings.wheelInvert).join(",");
        await callRobot("drive", drive, { cmd: "run", dirs, speed: settings.wheelSpeed, ms: settings.driveMs });
        setStatus({ severity: "success", text: t("status.driveDone") });
      } catch (e) {
        fail(e);
      } finally {
        setBusy(null);
      }
    },
    [drive, busy, settings, t, fail],
  );

  // stop is never blocked by a running move: it goes straight to the wheels
  const stopWheels = useCallback(async () => {
    if (!drive) return;
    try {
      await callRobot("drive", drive, { cmd: "stop" });
      setStatus({ severity: "success", text: t("status.stopped") });
    } catch (e) {
      fail(e);
    }
  }, [drive, t, fail]);

  // the distance sensor (HC-SR04 on the camera's motor board), read every second while that board is set; not
  // during a camera move (the board answers one request at a time)
  const cameraMoving = busy === "camera";
  useEffect(() => {
    if (!motors || cameraMoving) return;
    let alive = true;
    const read = () =>
      callRobot("motors", motors, { cmd: "distance" })
        .then((r) => r.text())
        .then((text) => alive && setDistance(text.trim()))
        .catch(() => alive && setDistance(null));
    const timer = setInterval(read, DISTANCE_EVERY_MS);
    void read();
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [motors, cameraMoving]);

  const takePhoto = useCallback(async () => {
    if (!cam || busy) return;
    setBusy("photo");
    setStatus({ severity: "info", text: t("status.capturing") });
    try {
      // the frame on screen when the live picture runs, otherwise a fresh one from the camera
      const blob = streamOn && frames.blob && !frames.down ? frames.blob : await (await callRobot("cam", cam, { cmd: "capture" })).blob();
      if (!blob.type.startsWith("image/")) throw new RobotCallError("failed");
      const photo: Photo = { id: crypto.randomUUID(), url: URL.createObjectURL(blob), takenAt: new Date(), panDeg: pose.pan, heightSteps: pose.height };
      setPhotos((ps) => [photo, ...ps]);
      setStatus({ severity: "success", text: t("status.captured") });
    } catch (e) {
      fail(e);
    } finally {
      setBusy(null);
    }
  }, [cam, busy, pose, streamOn, frames, t, fail]);

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

  // keyboard shortcuts, except while typing in a field
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.key === " ") {
        e.preventDefault();
        void takePhoto();
        return;
      }
      if (e.key === "Escape" || e.key.toLowerCase() === "x") {
        void stopWheels();
        return;
      }
      const k = KEYS[e.key] ?? KEYS[e.key.toLowerCase()];
      if (!k) return;
      e.preventDefault();
      if (k.kind === "camera") void moveCamera(k.arrow);
      else void driveTo(DRIVE_OF[k.arrow]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [takePhoto, moveCamera, driveTo, stopWheels]);

  const cameraLabels = { up: t("camera.up"), down: t("camera.down"), left: t("camera.left"), right: t("camera.right") };
  const driveLabels = { up: t("drive.forward"), down: t("drive.back"), left: t("drive.left"), right: t("drive.right") };
  const showStream = cam && streamOn && frames.url && !frames.down;

  return (
    <Box sx={{ display: "grid", gap: 4 }}>
      <Box sx={{ display: "grid", gap: 4, gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 3fr) minmax(320px, 2fr)" }, alignItems: "start" }}>
        {/* ---- live camera ---- */}
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
              <Button variant="contained" startIcon={<PhotoCameraOutlined />} onClick={takePhoto} disabled={!cam || busy !== null} data-testid="robot-capture">
                {t("live.capture")}
              </Button>
            </Box>
          </Box>
          <Box sx={{ position: "relative", aspectRatio: "4 / 3", bgcolor: "grey.900", borderRadius: 2, overflow: "hidden", display: "grid", placeItems: "center" }}>
            {showStream ? (
              <>
                {/* the latest frame (object URL of a /capture); a new one replaces it as soon as it arrives */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={frames.url!} alt={t("live.title")} data-testid="robot-frame" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                <Typography variant="caption" sx={{ position: "absolute", right: 8, bottom: 6, color: "common.white", textShadow: "0 0 3px black" }}>
                  {t("live.fps", { fps: frames.fps.toFixed(1) })}
                </Typography>
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

        {/* ---- controls ---- */}
        <Box sx={{ display: "grid", gap: 4 }}>
          <Paper sx={{ p: 4 }}>
            <Typography variant="h3" sx={{ mb: 1 }}>
              {t("camera.title")}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {t("camera.hint", { deg: settings.stepDeg, steps: settings.liftSteps, pan: Math.round(pose.pan), height: pose.height })}
            </Typography>
            {motors && (
              <Typography variant="subtitle2" sx={{ mb: 2 }} data-testid="robot-distance">
                {t("camera.distance", { value: distance ?? "—" })}
              </Typography>
            )}
            <ArrowPad
              testId="robot-camera-pad"
              labels={cameraLabels}
              disabled={!motors}
              busy={busy !== null}
              onPress={moveCamera}
              centre={
                <Tooltip title={t("camera.zero")}>
                  <IconButton onClick={() => setPose({ pan: 0, height: 0 })} aria-label={t("camera.zero")}>
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
              {t("drive.hint", { ms: settings.driveMs, speed: settings.wheelSpeed })}
            </Typography>
            <ArrowPad
              testId="robot-drive-pad"
              labels={driveLabels}
              disabled={!drive}
              busy={busy !== null}
              onPress={(a) => void driveTo(DRIVE_OF[a])}
              centre={
                <Tooltip title={t("drive.stop")}>
                  <span>
                    <IconButton color="error" onClick={() => void stopWheels()} disabled={!drive} aria-label={t("drive.stop")}>
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
