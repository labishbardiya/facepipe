import Link from "next/link";
import { ArrowRight, ShieldCheck, Zap, ScanFace } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white selection:bg-white/20">
      {/* Navigation */}
      <nav className="fixed top-0 w-full border-b border-white/10 bg-black/50 backdrop-blur-md z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
            <ScanFace className="w-6 h-6 text-white" />
            FacePipe
          </div>
          <Link
            href="/dashboard"
            className="text-sm font-medium bg-white text-black px-4 py-2 rounded-full hover:bg-neutral-200 transition-colors"
          >
            Go to Dashboard
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative pt-32 pb-16 sm:pt-40 sm:pb-24 lg:pb-32 overflow-hidden">
        {/* Glowing background effects */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/5 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 text-sm text-neutral-300 mb-8 backdrop-blur-md">
            <span className="flex h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
            Production-Ready Biometrics
          </div>
          
          <h1 className="text-5xl sm:text-7xl font-bold tracking-tight mb-8 leading-[1.1]">
            Secure Face Recognition. <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-neutral-200 to-neutral-600">
              Built for the Enterprise.
            </span>
          </h1>
          
          <p className="max-w-2xl mx-auto text-lg sm:text-xl text-neutral-400 mb-10 leading-relaxed">
            Bridge the gap between academic face recognition and enterprise deployment. 
            FacePipe handles security, deepfake detection, image quality, and high-speed vector search out of the box.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/dashboard"
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-white text-black px-8 py-4 rounded-full font-medium hover:bg-neutral-200 hover:scale-105 active:scale-95 transition-all duration-200"
            >
              Launch Dashboard
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Feature Grid */}
        <div className="max-w-7xl mx-auto px-6 mt-32">
          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard 
              icon={<ShieldCheck className="w-6 h-6 text-white" />}
              title="Decision Fusion Engine"
              description="Combines liveness, deepfake probability, and embedding similarity into a single cryptographic threshold."
            />
            <FeatureCard 
              icon={<ScanFace className="w-6 h-6 text-white" />}
              title="Anti-Spoofing & Liveness"
              description="Stop presentation attacks. FacePipe rejects photos of photos and 3D masks before recognition even runs."
            />
            <FeatureCard 
              icon={<Zap className="w-6 h-6 text-white" />}
              title="High-Speed Vector Search"
              description="Millisecond latency across millions of encrypted identities using optimized FAISS indexes."
            />
          </div>
        </div>
      </main>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="group relative p-8 rounded-3xl border border-white/10 bg-white/5 hover:bg-white/[0.07] transition-colors overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="relative z-10">
        <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center mb-6">
          {icon}
        </div>
        <h3 className="text-xl font-semibold mb-3">{title}</h3>
        <p className="text-neutral-400 leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  );
}
