"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TextField from "@mui/material/TextField";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import MapOutlined from "@mui/icons-material/MapOutlined";
import EditNoteOutlined from "@mui/icons-material/EditNoteOutlined";
import AssignmentIndOutlined from "@mui/icons-material/AssignmentIndOutlined";
import DeleteOutlined from "@mui/icons-material/DeleteOutlined";
import AddTaskOutlined from "@mui/icons-material/AddTaskOutlined";
import { KpiCard, KpiGrid } from "@/components/common/KpiCard";
import { ConfirmDialog, useAdminAction } from "@/components/admin/common";
import { Link } from "@/i18n/routing";
import { useFormat } from "@/lib/useFormat";
import { assignTasks, createTasksFromTargets, deleteTasks, setTaskStatus } from "@/app/[locale]/(app)/sarcini/actions";
import type { Target, TaskKind, TaskRow, TaskStatus } from "@/lib/tasks";

export type TaskRole = "uat_admin" | "inspector" | "viewer";
export interface Assignable {
  id: string;
  email: string;
  name: string | null;
}

const STATUSES: TaskStatus[] = ["open", "in_progress", "done", "cancelled"];
const STATUS_COLOR: Record<TaskStatus, "default" | "info" | "success" | "warning"> = {
  open: "warning",
  in_progress: "info",
  done: "success",
  cancelled: "default",
};

type Titled = { kind: TaskKind; row_id: string | null; vineyard_id: string | null; gap_length_m: number | null };

function useTaskTitle() {
  const t = useTranslations("tasks");
  const f = useFormat();
  return (x: Titled) => {
    if (x.kind === "gap") return x.gap_length_m ? t("titleGap", { m: f.m(x.gap_length_m, 1), row: x.row_id ?? "?" }) : t("titleGapNoLen", { row: x.row_id ?? "?" });
    if (x.kind === "missing") return t("titleMissing", { row: x.row_id ?? "?" });
    if (x.kind === "waste") return t("titleWaste", { block: x.vineyard_id ?? "?" });
    return t("titleOther");
  };
}

function AssignFields({
  members,
  value,
  onChange,
}: {
  members: Assignable[];
  value: { assignee: string; dueDate: string; priority: number };
  onChange: (v: { assignee: string; dueDate: string; priority: number }) => void;
}) {
  const t = useTranslations("tasks");
  return (
    <>
      <TextField select label={t("fieldAssignee")} value={value.assignee} onChange={(e) => onChange({ ...value, assignee: e.target.value })} fullWidth>
        <MenuItem value="">
          <em>{t("unassigned")}</em>
        </MenuItem>
        {members.map((m) => (
          <MenuItem key={m.id} value={m.id}>
            {m.name ? `${m.name} · ${m.email}` : m.email}
          </MenuItem>
        ))}
      </TextField>
      <TextField
        type="date"
        label={t("fieldDue")}
        value={value.dueDate}
        onChange={(e) => onChange({ ...value, dueDate: e.target.value })}
        fullWidth
        slotProps={{ inputLabel: { shrink: true } }}
      />
      <TextField select label={t("fieldPriority")} value={value.priority} onChange={(e) => onChange({ ...value, priority: Number(e.target.value) })} fullWidth>
        {[1, 2, 3].map((p) => (
          <MenuItem key={p} value={p}>
            {t(`priority.${p}`)}
          </MenuItem>
        ))}
      </TextField>
    </>
  );
}

export function TasksPanel({
  role,
  meId,
  tasks,
  targets,
  members,
  adminApi,
}: {
  role: TaskRole;
  meId: string;
  tasks: TaskRow[];
  targets: Target[];
  members: Assignable[];
  adminApi: boolean;
}) {
  const t = useTranslations("tasks");
  const tc = useTranslations("superAdmin.common");
  const f = useFormat();
  const title = useTaskTitle();
  const { run, pending, snackbar } = useAdminAction();
  const isAdmin = role === "uat_admin";

  const [scope, setScope] = useState<"all" | "mine">(role === "inspector" ? "mine" : "all");
  const [status, setStatus] = useState<"active" | TaskStatus | "any">("active");
  const [selTasks, setSelTasks] = useState<Set<number>>(new Set());
  const [selTargets, setSelTargets] = useState<Set<string>>(new Set());
  const [creating, setCreating] = useState(false);
  const [assigning, setAssigning] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [editing, setEditing] = useState<TaskRow | null>(null);
  const [assign, setAssign] = useState({ assignee: "", dueDate: "", priority: 2 });
  const [statusDraft, setStatusDraft] = useState<{ status: TaskStatus; note: string }>({ status: "in_progress", note: "" });

  // the cards follow the All / Mine switch, like the table under them
  const counts = useMemo(() => {
    const inScope = scope === "all" ? tasks : tasks.filter((x) => x.assignee === meId);
    return {
      open: inScope.filter((x) => x.status === "open").length,
      in_progress: inScope.filter((x) => x.status === "in_progress").length,
      done: inScope.filter((x) => x.status === "done").length,
      unassigned: tasks.filter((x) => !x.assignee && x.status !== "done" && x.status !== "cancelled").length,
    };
  }, [tasks, scope, meId]);
  const visible = useMemo(
    () =>
      tasks.filter(
        (x) =>
          (scope === "all" || x.assignee === meId) &&
          (status === "any" || (status === "active" ? x.status === "open" || x.status === "in_progress" : x.status === status)),
      ),
    [tasks, scope, status, meId],
  );
  const canUpdate = (x: TaskRow) => isAdmin || (role === "inspector" && x.assignee === meId);
  const toggle = <T,>(set: Set<T>, v: T) => {
    const n = new Set(set);
    if (n.has(v)) n.delete(v);
    else n.add(v);
    return n;
  };
  const targetKey = (x: Target) => `${x.survey_id}/${x.target_id}`;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <KpiGrid min={170}>
        <KpiCard label={t("kpiOpen")} value={f.int(counts.open)} tone="warning" />
        <KpiCard label={t("kpiInProgress")} value={f.int(counts.in_progress)} tone="primary" />
        <KpiCard label={t("kpiDone")} value={f.int(counts.done)} />
        {scope === "all" && <KpiCard label={t("kpiUnassigned")} value={f.int(counts.unassigned)} />}
      </KpiGrid>

      {role === "viewer" && <Alert severity="info">{t("readOnly")}</Alert>}
      {isAdmin && !adminApi && <Alert severity="warning">{t("noKey")}</Alert>}

      {isAdmin && (
        <Card>
          <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", gap: 3, flexWrap: "wrap" }}>
            <Box sx={{ flex: 1, minWidth: 260 }}>
              <Typography variant="h3">{t("targetsTitle", { n: targets.length })}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {targets.length ? t("targetsHint") : t("noTargets")}
              </Typography>
            </Box>
            <Button
              variant="contained"
              startIcon={<AddTaskOutlined />}
              disabled={selTargets.size === 0 || pending}
              onClick={() => {
                setAssign({ assignee: "", dueDate: "", priority: 2 });
                setCreating(true);
              }}
            >
              {t("createTasks", { n: selTargets.size })}
            </Button>
          </Box>
          {targets.length > 0 && (
            <Box sx={{ overflowX: "auto" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selTargets.size === targets.length}
                        indeterminate={selTargets.size > 0 && selTargets.size < targets.length}
                        onChange={(e) => setSelTargets(e.target.checked ? new Set(targets.map(targetKey)) : new Set())}
                      />
                    </TableCell>
                    <TableCell>{t("colTask")}</TableCell>
                    <TableCell>{t("colPlace")}</TableCell>
                    <TableCell align="right" />
                  </TableRow>
                </TableHead>
                <TableBody>
                  {targets.map((x) => (
                    <TableRow key={targetKey(x)} hover selected={selTargets.has(targetKey(x))}>
                      <TableCell padding="checkbox">
                        <Checkbox checked={selTargets.has(targetKey(x))} onChange={() => setSelTargets((s) => toggle(s, targetKey(x)))} />
                      </TableCell>
                      <TableCell>
                        <Chip size="small" variant="outlined" label={t(`kind.${x.kind}`)} sx={{ mr: 2 }} />
                        {title(x)}
                      </TableCell>
                      <TableCell>
                        {[x.vineyard_id, x.row_id].filter(Boolean).join(" · ")}
                        {x.route_order ? (
                          <Typography component="span" variant="caption" color="text.secondary">
                            {` · ${x.target_id} · ${t("route", { n: x.route_order })}`}
                          </Typography>
                        ) : null}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title={t("showOnMap")}>
                          <IconButton size="small" component={Link} href={`/harta?tinta=${x.target_id}`}>
                            <MapOutlined fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          )}
        </Card>
      )}

      <Card>
        <Box sx={{ px: 5, py: 4, display: "flex", alignItems: "center", gap: 3, flexWrap: "wrap" }}>
          <ToggleButtonGroup size="small" exclusive value={scope} onChange={(_, v) => v && setScope(v)}>
            <ToggleButton value="all">{t("filterAll")}</ToggleButton>
            <ToggleButton value="mine">{t("filterMine")}</ToggleButton>
          </ToggleButtonGroup>
          <TextField select size="small" label={t("filterStatus")} value={status} onChange={(e) => setStatus(e.target.value as typeof status)} sx={{ minWidth: 190 }}>
            <MenuItem value="active">{`${t("status.open")} + ${t("status.in_progress")}`}</MenuItem>
            {STATUSES.map((s) => (
              <MenuItem key={s} value={s}>
                {t(`status.${s}`)}
              </MenuItem>
            ))}
            <MenuItem value="any">{t("filterAll")}</MenuItem>
          </TextField>
          <Box sx={{ flex: 1 }} />
          {isAdmin && selTasks.size > 0 && (
            <>
              <Typography variant="body2" color="text.secondary">
                {t("selected", { n: selTasks.size })}
              </Typography>
              <Button
                startIcon={<AssignmentIndOutlined />}
                onClick={() => {
                  setAssign({ assignee: "", dueDate: "", priority: 2 });
                  setAssigning(true);
                }}
                disabled={pending}
              >
                {t("assign")}
              </Button>
              <Button color="error" startIcon={<DeleteOutlined />} onClick={() => setDeleting(true)} disabled={pending}>
                {t("delete")}
              </Button>
            </>
          )}
        </Box>
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                {isAdmin && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={visible.length > 0 && visible.every((x) => selTasks.has(x.id))}
                      onChange={(e) => setSelTasks(e.target.checked ? new Set(visible.map((x) => x.id)) : new Set())}
                    />
                  </TableCell>
                )}
                <TableCell>{t("colTask")}</TableCell>
                <TableCell>{t("colPlace")}</TableCell>
                <TableCell>{t("colAssignee")}</TableCell>
                <TableCell>{t("colStatus")}</TableCell>
                <TableCell>{t("colDue")}</TableCell>
                <TableCell>{t("colPriority")}</TableCell>
                <TableCell>{t("colUpdated")}</TableCell>
                <TableCell align="right">{t("colActions")}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {visible.length === 0 && (
                <TableRow>
                  <TableCell colSpan={isAdmin ? 9 : 8}>
                    <Typography color="text.secondary">{t("empty")}</Typography>
                  </TableCell>
                </TableRow>
              )}
              {visible.map((x) => (
                <TableRow key={x.id} hover selected={selTasks.has(x.id)}>
                  {isAdmin && (
                    <TableCell padding="checkbox">
                      <Checkbox checked={selTasks.has(x.id)} onChange={() => setSelTasks((s) => toggle(s, x.id))} />
                    </TableCell>
                  )}
                  <TableCell>
                    <Chip size="small" variant="outlined" label={t(`kind.${x.kind}`)} sx={{ mr: 2 }} />
                    <Typography component="span" variant="body2" sx={{ fontWeight: 600 }}>
                      {title(x)}
                    </Typography>
                    {x.resolution_note && (
                      <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 0.5 }}>
                        “{x.resolution_note}”
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>{[x.vineyard_id, x.row_id].filter(Boolean).join(" · ") || "—"}</TableCell>
                  <TableCell>{x.assignee_email ?? <em>{t("unassigned")}</em>}</TableCell>
                  <TableCell>
                    <Chip size="small" color={STATUS_COLOR[x.status]} label={t(`status.${x.status}`)} />
                  </TableCell>
                  <TableCell>{x.due_date ? f.date(x.due_date) : "—"}</TableCell>
                  <TableCell>{t(`priority.${x.priority}`)}</TableCell>
                  <TableCell>{f.date(x.updated_at)}</TableCell>
                  <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                    {x.target_id && (
                      <Tooltip title={t("showOnMap")}>
                        <IconButton size="small" component={Link} href={`/harta?tinta=${x.target_id}`}>
                          <MapOutlined fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    {canUpdate(x) && (
                      <Tooltip title={t("update")}>
                        <IconButton
                          size="small"
                          onClick={() => {
                            setStatusDraft({ status: x.status === "open" ? "in_progress" : x.status, note: x.resolution_note ?? "" });
                            setEditing(x);
                          }}
                        >
                          <EditNoteOutlined fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      </Card>

      {/* create from targets */}
      <Dialog open={creating} onClose={() => setCreating(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t("createTitle", { n: selTargets.size })}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <AssignFields members={members} value={assign} onChange={setAssign} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreating(false)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending}
            onClick={() =>
              run(
                () => createTasksFromTargets({ targetIds: [...selTargets], assignee: assign.assignee || null, dueDate: assign.dueDate || null, priority: assign.priority }),
                () => {
                  setCreating(false);
                  setSelTargets(new Set());
                },
              )
            }
          >
            {t("createTasks", { n: selTargets.size })}
          </Button>
        </DialogActions>
      </Dialog>

      {/* assign selected tasks */}
      <Dialog open={assigning} onClose={() => setAssigning(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t("assignTitle", { n: selTasks.size })}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <AssignFields members={members} value={assign} onChange={setAssign} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAssigning(false)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending}
            onClick={() =>
              run(
                () => assignTasks({ ids: [...selTasks], assignee: assign.assignee || null, dueDate: assign.dueDate || null, priority: assign.priority }),
                () => {
                  setAssigning(false);
                  setSelTasks(new Set());
                },
              )
            }
          >
            {t("assign")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* status update (admin or assignee) */}
      <Dialog open={editing !== null} onClose={() => setEditing(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{editing ? `${t("statusTitle")} · ${title(editing)}` : ""}</DialogTitle>
        <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <TextField select label={t("fieldStatus")} value={statusDraft.status} onChange={(e) => setStatusDraft({ ...statusDraft, status: e.target.value as TaskStatus })} fullWidth>
            {STATUSES.filter((s) => isAdmin || s !== "cancelled").map((s) => (
              <MenuItem key={s} value={s}>
                {t(`status.${s}`)}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            label={t("fieldNote")}
            value={statusDraft.note}
            onChange={(e) => setStatusDraft({ ...statusDraft, note: e.target.value })}
            multiline
            minRows={3}
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditing(null)}>{tc("cancel")}</Button>
          <Button
            variant="contained"
            disabled={pending}
            onClick={() => editing && run(() => setTaskStatus({ id: editing.id, ...statusDraft }), () => setEditing(null))}
          >
            {t("update")}
          </Button>
        </DialogActions>
      </Dialog>

      <ConfirmDialog
        open={deleting}
        text={t("deleteConfirm", { n: selTasks.size })}
        onConfirm={() => run(() => deleteTasks([...selTasks]), () => setSelTasks(new Set()))}
        onClose={() => setDeleting(false)}
      />
      {snackbar}
    </Box>
  );
}
