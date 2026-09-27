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
import { parseBoardUrl, type DriveCmd } from "@/lib/robot/commands";
import { ArrowPad, type Arrow } from "./ArrowPad";
import { PhotoGallery, type Photo } from "./PhotoGallery";
import { SettingsCard } from "./SettingsCard";
import { callRobot, RobotCallError, type RobotError } from "./robotApi";
import { useRobotSettings } from "./useRobotSettings";

/** the ESP32-CAM serves one request at a time: the stream pauses this long before a capture */
const STREAM_PAUSE_MS = 300;
const DRIVE_OF: Record<Arrow, DriveCmd> = { up: "forward", down: "back", left: "left", right: "right" };
/** camera arrows → stepper and direction: motor 1 pans (0 = right), motor 2 tilts (0 = up) */
const CAMERA_OF: Record<Arrow, { motor: 1 | 2; dir: 0 | 1; axis: "pan" | "tilt"; sign: 1 | -1 }> = {
  right: { motor: 1, dir: 0, axis: "pan", sign: 1 },
  left: { motor: 1, dir: 1, axis: "pan", sign: -1 },
  up: { motor: 2, dir: 0, axis: "tilt", sign: 1 },
  down: { motor: 2, dir: 1, axis: "tilt", sign: -1 },
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
  const [streamPaused, setStreamPaused] = useState(false);
  // the address whose stream failed to load (a new address gets a fresh try)
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const streamFailed = failedUrl !== null && failedUrl === cam;
  const [busy, setBusy] = useState<"camera" | "drive" | "photo" | null>(null);
  const [status, setStatus] = useState<Status>(null);
  const [angles, setAngles] = useState({ pan: 0, tilt: 0 });
  const [photos, setPhotos] = useState<Photo[]>([]);
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
      const steps = Math.max(1, Math.round((settings.stepDeg / 360) * settings.stepsPerRev));
      setBusy("camera");
      setStatus({ severity: "info", text: t("status.cameraMoving", { dir: t(`camera.${a}`) }) });
      try {
        await callRobot("motors", motors, { cmd: "move", motor: m.motor, dir: m.dir, speed: settings.speedPps, steps });
        const deg = (steps * 360) / settings.stepsPerRev;
        setAngles((s) => ({ ...s, [m.axis]: s[m.axis] + m.sign * deg }));
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
    async (cmd: DriveCmd) => {
      if (!drive || (busy && cmd !== "stop")) return;
      setBusy(cmd === "stop" ? busy : "drive");
      setStatus({ severity: "info", text: t("status.driving", { dir: t(`drive.${cmd}`) }) });
      try {
        await callRobot("drive", drive, { cmd, ms: settings.driveMs });
        setStatus({ severity: "success", text: t("status.driveDone") });
      } catch (e) {
        fail(e);
      } finally {
        if (cmd !== "stop") setBusy(null);
      }
    },
    [drive, busy, settings.driveMs, t, fail],
  );

  const takePhoto = useCallback(async () => {
    if (!cam || busy) return;
    setBusy("photo");
    setStatus({ severity: "info", text: t("status.capturing") });
    setStreamPaused(true);
    try {
      await new Promise((r) => setTimeout(r, STREAM_PAUSE_MS));
      const res = await callRobot("cam", cam, { cmd: "capture" });
      const blob = await res.blob();
      if (!blob.type.startsWith("image/")) throw new RobotCallError("failed");
      const photo: Photo = { id: crypto.randomUUID(), url: URL.createObjectURL(blob), takenAt: new Date(), panDeg: angles.pan, tiltDeg: angles.tilt };
      setPhotos((ps) => [photo, ...ps]);
      setStatus({ severity: "success", text: t("status.captured") });
    } catch (e) {
      fail(e);
    } finally {
      setStreamPaused(false);
      setBusy(null);
    }
  }, [cam, busy, angles, t, fail]);

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
      const k = KEYS[e.key] ?? KEYS[e.key.toLowerCase()];
      if (!k) return;
      e.preventDefault();
      if (k.kind === "camera") void moveCamera(k.arrow);
      else void driveTo(DRIVE_OF[k.arrow]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [takePhoto, moveCamera, driveTo]);

  const cameraLabels = { up: t("camera.up"), down: t("camera.down"), left: t("camera.left"), right: t("camera.right") };
  const driveLabels = { up: t("drive.forward"), down: t("drive.back"), left: t("drive.left"), right: t("drive.right") };
  const showStream = cam && streamOn && !streamPaused && !streamFailed;

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
              <Button size="small" onClick={() => { setFailedUrl(null); setStreamOn((v) => !v); }} disabled={!cam}>
                {streamOn ? t("live.stop") : t("live.start")}
              </Button>
              <Button variant="contained" startIcon={<PhotoCameraOutlined />} onClick={takePhoto} disabled={!cam || busy !== null} data-testid="robot-capture">
                {t("live.capture")}
              </Button>
            </Box>
          </Box>
          <Box sx={{ position: "relative", aspectRatio: "4 / 3", bgcolor: "grey.900", borderRadius: 2, overflow: "hidden", display: "grid", placeItems: "center" }}>
            {showStream ? (
              // the MJPEG stream straight from the camera (an <img> needs no CORS)
              // eslint-disable-next-line @next/next/no-img-element
              <img src={`${cam}/stream`} alt={t("live.title")} onError={() => setFailedUrl(cam)} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
            ) : (
              <Box sx={{ color: "grey.400", textAlign: "center", px: 4 }}>
                <VideocamOffOutlined fontSize="large" />
                <Typography variant="body2">
                  {!cam ? t("live.noAddress") : streamFailed ? t("live.failed", { url: cam }) : streamPaused ? t("live.capturing") : t("live.off")}
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
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {t("camera.hint", { deg: settings.stepDeg, pan: Math.round(angles.pan), tilt: Math.round(angles.tilt) })}
            </Typography>
            <ArrowPad
              testId="robot-camera-pad"
              labels={cameraLabels}
              disabled={!motors}
              busy={busy !== null}
              onPress={moveCamera}
              centre={
                <Tooltip title={t("camera.zero")}>
                  <IconButton onClick={() => setAngles({ pan: 0, tilt: 0 })} aria-label={t("camera.zero")}>
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
              {t("drive.hint", { ms: settings.driveMs })}
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
                    <IconButton color="error" onClick={() => void driveTo("stop")} disabled={!drive} aria-label={t("drive.stop")}>
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
