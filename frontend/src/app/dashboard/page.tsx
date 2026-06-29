"use client";

import { useState, useRef } from "react";
import { UploadCloud, ShieldCheck, Camera, Video, Settings2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function VerificationDashboard() {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [mode, setMode] = useState<"photo" | "video">("photo");
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
    setResult(null);
    setError(null);
  };

  const handleVerify = async () => {
    if (!imagePreview) return;
    setIsVerifying(true);
    setResult(null);
    setError(null);
    
    // Extract base64 without the data:image/jpeg;base64, prefix
    const base64Data = imagePreview.split(",")[1];

    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data, mode }),
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Verification failed");
      
      if (data.face_count === 0) {
        setError("No face detected in the image.");
      } else {
        setResult(data.faces[0]);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <div className="relative space-y-12 pb-24">
      {/* Abstract Background Elements */}
      <div className="fixed inset-0 z-[-1] pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-violet-600/10 blur-[120px]" />
        <div className="absolute top-[40%] left-[60%] w-[30%] h-[30%] rounded-full bg-cyan-600/10 blur-[120px]" />
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <h1 className="text-6xl font-serif tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-br from-white to-white/60 drop-shadow-sm">
          Identity Verification
        </h1>
        <p className="text-lg text-neutral-400 max-w-xl font-light">
          Authenticate instantly against the secure neural database. Select your input mode below to optimize the fusion engine.
        </p>
      </motion.div>

      {/* Mode Selector */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="inline-flex p-1 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md"
      >
        <button
          onClick={() => setMode("photo")}
          className={`relative flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium transition-colors ${mode === "photo" ? "text-white" : "text-neutral-400 hover:text-white"}`}
        >
          {mode === "photo" && (
            <motion.div layoutId="mode-bg" className="absolute inset-0 bg-white/10 rounded-xl" />
          )}
          <Camera className="w-4 h-4 relative z-10" />
          <span className="relative z-10">Photo Mode (No Tracking)</span>
        </button>
        <button
          onClick={() => setMode("video")}
          className={`relative flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium transition-colors ${mode === "video" ? "text-white" : "text-neutral-400 hover:text-white"}`}
        >
          {mode === "video" && (
            <motion.div layoutId="mode-bg" className="absolute inset-0 bg-white/10 rounded-xl" />
          )}
          <Video className="w-4 h-4 relative z-10" />
          <span className="relative z-10">Video Mode (Strict)</span>
        </button>
      </motion.div>

      {/* Upload Zone */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        onClick={() => fileInputRef.current?.click()}
        className="relative group p-8 rounded-[2rem] border border-white/10 bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/20 transition-all cursor-pointer flex flex-col items-center justify-center min-h-[400px] text-center overflow-hidden shadow-2xl backdrop-blur-sm"
      >
        <input 
          type="file" 
          accept="image/*" 
          className="hidden" 
          ref={fileInputRef}
          onChange={handleImageSelect} 
        />
        
        <AnimatePresence mode="wait">
          {imagePreview ? (
            <motion.img 
              key="preview"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
              src={imagePreview} 
              alt="Preview" 
              className="absolute inset-0 w-full h-full object-contain p-6 z-10" 
            />
          ) : (
            <motion.div 
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center"
            >
              <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500 ease-out border border-white/5">
                <UploadCloud className="w-8 h-8 text-blue-400" />
              </div>
              <h3 className="text-xl font-medium mb-2 font-serif tracking-wide text-white/90">Drop image here</h3>
              <p className="text-sm text-neutral-400 max-w-[250px] font-light">
                Click to browse your file system or drag a selfie here.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Action Bar */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between p-6 rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-xl gap-4"
      >
        <div className="flex items-center gap-5">
          <div className="w-14 h-14 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/10 flex items-center justify-center border border-blue-500/20 shadow-inner">
            <ShieldCheck className="w-7 h-7 text-blue-400" />
          </div>
          <div>
            <h3 className="font-medium text-white tracking-wide text-lg">Analysis Engine Ready</h3>
            <p className="text-sm text-neutral-400 font-light flex items-center gap-2">
              <Settings2 className="w-3 h-3" />
              {mode === "photo" ? "Tracking bypassed" : "Strict tracking enabled"}
            </p>
          </div>
        </div>
        <button
          onClick={handleVerify}
          disabled={isVerifying || !imagePreview}
          className="relative overflow-hidden bg-white text-black px-10 py-4 rounded-full font-medium hover:bg-neutral-200 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed group shadow-[0_0_40px_rgba(255,255,255,0.1)] hover:shadow-[0_0_60px_rgba(255,255,255,0.2)]"
        >
          {isVerifying ? (
            <span className="flex items-center gap-3">
              <span className="w-5 h-5 rounded-full border-2 border-black/20 border-t-black animate-spin" />
              Verifying...
            </span>
          ) : (
            <span className="tracking-wide">Authenticate</span>
          )}
        </button>
      </motion.div>

      {/* Results Area */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-5 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 backdrop-blur-md"
          >
            {error}
          </motion.div>
        )}

        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="p-8 rounded-[2rem] border border-white/10 bg-white/5 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 backdrop-blur-xl shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-white/5 to-transparent rounded-bl-full pointer-events-none" />
              
              <div className="relative z-10">
                <p className="text-neutral-400 mb-2 font-light uppercase tracking-widest text-xs">Identity Match</p>
                <h2 className="text-5xl font-serif text-white tracking-tight">{result.identity || "Unknown Entity"}</h2>
              </div>
              <div className={`relative z-10 px-6 py-3 rounded-xl font-medium tracking-wide shadow-lg ${result.is_recognized ? 'bg-green-500/10 text-green-400 border border-green-500/20 shadow-green-500/10' : 'bg-red-500/10 text-red-400 border border-red-500/20 shadow-red-500/10'}`}>
                {result.decision}
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <MetricCard 
                title="Confidence Score" 
                value={`${(result.confidence * 100).toFixed(1)}%`} 
                status={result.confidence > 0.4 ? 'success' : 'danger'} 
                delay={0.1}
              />
              <MetricCard 
                title="Liveness Assessment" 
                value={result.is_live ? "Live" : "Spoof"} 
                status={result.is_live ? 'success' : 'danger'} 
                delay={0.2}
              />
              <MetricCard 
                title="Deepfake Check" 
                value={result.is_real ? "Real" : "Fake"} 
                status={result.is_real ? 'success' : 'danger'} 
                delay={0.3}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MetricCard({ title, value, status, delay = 0 }: { title: string, value: string, status: 'success' | 'warning' | 'danger', delay?: number }) {
  const colors = {
    success: "text-green-400 bg-green-400/10 border-green-400/20 shadow-green-400/5",
    warning: "text-amber-400 bg-amber-400/10 border-amber-400/20 shadow-amber-400/5",
    danger: "text-red-400 bg-red-400/10 border-red-400/20 shadow-red-400/5",
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={`p-6 rounded-3xl border backdrop-blur-md ${colors[status]} flex flex-col justify-between min-h-[140px] relative overflow-hidden group`}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <p className="text-sm font-light tracking-wide opacity-80">{title}</p>
      <h3 className="text-4xl font-serif tracking-tight mt-4">{value}</h3>
    </motion.div>
  );
}
