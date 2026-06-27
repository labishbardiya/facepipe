"use client";

import { useState, useRef } from "react";
import { UploadCloud, ShieldCheck, Camera } from "lucide-react";

export default function VerificationDashboard() {
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
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
        body: JSON.stringify({ image: base64Data }),
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
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">Face Recognition</h1>
        <p className="text-neutral-400">Upload a live selfie to recognize identity against the secure database.</p>
      </div>

      {/* Upload Zone */}
      <div 
        onClick={() => fileInputRef.current?.click()}
        className="relative group p-8 rounded-3xl border-2 border-dashed border-white/10 bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/20 transition-all cursor-pointer flex flex-col items-center justify-center min-h-[350px] text-center overflow-hidden"
      >
        <input 
          type="file" 
          accept="image/*" 
          className="hidden" 
          ref={fileInputRef}
          onChange={handleImageSelect} 
        />
        
        {imagePreview ? (
          <img src={imagePreview} alt="Preview" className="absolute inset-0 w-full h-full object-contain p-4 z-10" />
        ) : (
          <>
            <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <Camera className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-medium mb-2">Live Selfie</h3>
            <p className="text-sm text-neutral-400 max-w-[200px]">
              Click to browse or take a photo.
            </p>
          </>
        )}
      </div>

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center border border-blue-500/20">
            <ShieldCheck className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h3 className="font-medium text-white">Ready for Analysis</h3>
            <p className="text-sm text-neutral-400">Deepfake and liveness checks enabled.</p>
          </div>
        </div>
        <button
          onClick={handleVerify}
          disabled={isVerifying || !imagePreview}
          className="relative overflow-hidden bg-white text-black px-8 py-3 rounded-full font-medium hover:bg-neutral-200 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          {isVerifying ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full border-2 border-black/20 border-t-black animate-spin" />
              Processing...
            </span>
          ) : (
            "Verify Identity"
          )}
        </button>
      </div>

      {/* Results Area */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 animate-in fade-in zoom-in-95 duration-500">
          <div className="p-6 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-between">
            <div>
              <p className="text-neutral-400 mb-1">Identified As</p>
              <h2 className="text-3xl font-bold">{result.identity || "Unknown"}</h2>
            </div>
            <div className={`px-4 py-2 rounded-lg font-medium ${result.is_recognized ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
              {result.decision}
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <MetricCard title="Confidence Score" value={`${(result.confidence * 100).toFixed(1)}%`} status={result.confidence > 0.4 ? 'success' : 'danger'} />
            <MetricCard title="Liveness Assessment" value={result.is_live ? "Live" : "Spoof"} status={result.is_live ? 'success' : 'danger'} />
            <MetricCard title="Deepfake Check" value={result.is_real ? "Real" : "Fake"} status={result.is_real ? 'success' : 'danger'} />
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value, status }: { title: string, value: string, status: 'success' | 'warning' | 'danger' }) {
  const colors = {
    success: "text-green-400 bg-green-400/10 border-green-400/20",
    warning: "text-amber-400 bg-amber-400/10 border-amber-400/20",
    danger: "text-red-400 bg-red-400/10 border-red-400/20",
  };

  return (
    <div className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md flex flex-col">
      <h4 className="text-sm font-medium text-neutral-400 mb-2">{title}</h4>
      <div className="flex items-end gap-3 mt-auto">
        <span className="text-3xl font-bold tracking-tight">{value}</span>
      </div>
    </div>
  );
}
