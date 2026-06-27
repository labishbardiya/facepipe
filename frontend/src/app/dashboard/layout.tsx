import Link from "next/link";
import { ScanFace, UserPlus, Fingerprint, LayoutDashboard, Settings } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-black text-white flex">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-white/10 bg-black/50 p-6 flex flex-col hidden md:flex">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl tracking-tight mb-12">
          <ScanFace className="w-6 h-6 text-white" />
          FacePipe
        </Link>
        
        <div className="space-y-6 flex-1">
          <div className="space-y-1">
            <p className="px-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-2">
              Biometrics
            </p>
            <NavItem href="/dashboard" icon={<Fingerprint className="w-5 h-5" />} label="Verification" active={true} />
            <NavItem href="/dashboard/enroll" icon={<UserPlus className="w-5 h-5" />} label="Enrollment" />
          </div>

          <div className="space-y-1">
            <p className="px-3 text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-2">
              System
            </p>
            <NavItem href="/dashboard/metrics" icon={<LayoutDashboard className="w-5 h-5" />} label="Metrics" />
            <NavItem href="/dashboard/settings" icon={<Settings className="w-5 h-5" />} label="Settings" />
          </div>
        </div>
        
        <div className="pt-6 border-t border-white/10 mt-auto">
          <div className="flex items-center gap-3 px-3 py-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-neutral-700 to-neutral-500 border border-white/20" />
            <div className="text-sm">
              <p className="font-medium">System Admin</p>
              <p className="text-neutral-500 text-xs">admin@facepipe.ai</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 relative overflow-y-auto">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-white/[0.03] via-transparent to-transparent pointer-events-none" />
        <div className="relative z-10 p-8 max-w-6xl mx-auto min-h-full">
          {children}
        </div>
      </main>
    </div>
  );
}

function NavItem({ href, icon, label, active = false }: { href: string, icon: React.ReactNode, label: string, active?: boolean }) {
  return (
    <Link
      href={href}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
        active 
          ? "bg-white/10 text-white shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]" 
          : "text-neutral-400 hover:text-white hover:bg-white/5"
      }`}
    >
      {icon}
      {label}
    </Link>
  );
}
