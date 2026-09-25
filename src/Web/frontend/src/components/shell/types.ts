import type { Country } from "@/lib/uats";
import type { UatRole } from "@/lib/viewer";

/** What the client shell needs about the viewer (no geometry, no survey list). */
export interface ShellViewer {
  kind: "user" | "demo";
  email: string | null;
  name: string | null;
  role: UatRole | null;
  uat: { key: string; name: string; district: string | null; country: Country } | null;
}
