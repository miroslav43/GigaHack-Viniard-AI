import { AppShell } from "@/components/shell/AppShell";
import { getViewer } from "@/lib/viewer";

export default async function AppLayout({ children }: LayoutProps<"/[locale]">) {
  const viewer = await getViewer();
  return <AppShell viewer={viewer}>{children}</AppShell>;
}
