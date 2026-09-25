import { AdminShell } from "@/components/shell/AdminShell";
import type { ShellViewer } from "@/components/shell/types";
import { getViewer } from "@/lib/viewer";

export default async function AdminLayout({ children }: LayoutProps<"/[locale]">) {
  const v = await getViewer();
  const viewer: ShellViewer = { kind: v.kind, email: v.email, name: v.name, role: v.role, uat: null };
  return <AdminShell viewer={viewer}>{children}</AdminShell>;
}
