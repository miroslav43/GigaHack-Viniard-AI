import { AppShell } from "@/components/shell/AppShell";
import type { ShellViewer } from "@/components/shell/types";
import { getViewer } from "@/lib/viewer";

export default async function AppLayout({ children }: LayoutProps<"/[locale]">) {
  const v = await getViewer();
  const viewer: ShellViewer = {
    kind: v.kind,
    userId: v.userId,
    email: v.email,
    name: v.name,
    role: v.role,
    uat: v.uat && { key: v.uat.key, name: v.uat.name, district: v.uat.district, country: v.uat.country },
  };
  return <AppShell viewer={viewer}>{children}</AppShell>;
}
