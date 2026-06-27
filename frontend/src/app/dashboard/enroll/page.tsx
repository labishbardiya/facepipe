"use client";

import { useState, useRef } from "react";
import { UploadCloud, UserPlus, CheckCircle2 } from "lucide-react";

export default function EnrollmentDashboard() {
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [images, setImages] = useState<string[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImages(prev => [...prev, ev.target?.result as string]);
      };
      reader.readAsDataURL(file);
    });
  };

  const handleEnroll = async () => {
    if (!name || images.length === 0) {
      setError("Please provide a name and at least one image.");
      return;
    }
    
    setIsEnrolling(true);
    setError(null);
    setResult(null);

    // Extract base64
    const base64Images = images.map(img => img.split(",")[1]);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, images: base64Images }),
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Enrollment failed");
      
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsEnrolling(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">New Enrollment</h1>
        <p className="text-neutral-400">Register a new identity into the encrypted vector store.</p>
      </div>

      <div className="max-w-md">
        <label className="block text-sm font-medium text-neutral-300 mb-2">Identity Name</label>
        <input 
          type="text" 
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. John Doe"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-neutral-500 focus:outline-none focus:ring-2 focus:ring-white/20 transition-all"
        />
      </div>

      {/* Upload Zone */}
      <div 
        onClick={() => fileInputRef.current?.click()}
        className="relative group p-12 rounded-3xl border-2 border-dashed border-white/10 bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/20 transition-all cursor-pointer flex flex-col items-center justify-center min-h-[300px] text-center overflow-hidden"
      >
        <input 
          type="file" 
          accept="image/*" 
          multiple
          className="hidden" 
          ref={fileInputRef}
          onChange={handleImageSelect} 
        />
        
        {images.length > 0 ? (
          <div className="flex flex-wrap gap-4 justify-center z-10 relative">
            {images.map((img, i) => (
              <img key={i} src={img} alt={`Preview ${i}`} className="h-32 w-32 object-cover rounded-xl border border-white/20 shadow-lg" />
            ))}
          </div>
        ) : (
          <>
            <div className="w-20 h-20 rounded-3xl bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <UploadCloud className="w-10 h-10 text-white" />
            </div>
            <h3 className="text-xl font-medium mb-3">Upload Reference Images</h3>
            <p className="text-neutral-400 max-w-sm">
              Upload at least 3 high-quality, front-facing photos of the user. FacePipe will extract and average the templates.
            </p>
          </>
        )}
      </div>
      
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400">
          {error}
        </div>
      )}
      
      {result && result.success && (
        <div className="p-6 rounded-2xl bg-green-500/10 border border-green-500/20 text-green-400 flex items-center gap-4 animate-in fade-in zoom-in-95">
          <CheckCircle2 className="w-8 h-8" />
          <div>
            <h3 className="font-bold text-lg">Enrolled Successfully</h3>
            <p className="text-sm opacity-80">{result.message} ({result.embeddings_stored} embeddings stored).</p>
          </div>
        </div>
      )}

      <div className="flex justify-end">
        <button
          onClick={handleEnroll}
          disabled={isEnrolling || images.length === 0 || !name}
          className="relative overflow-hidden bg-white text-black px-8 py-3 rounded-full font-medium hover:bg-neutral-200 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed group/btn"
        >
          {isEnrolling ? (
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full border-2 border-black/20 border-t-black animate-spin" />
              Processing...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <UserPlus className="w-5 h-5" />
              Process Enrollment
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
