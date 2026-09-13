import { useCallback, useEffect, useRef, useState } from 'react';

interface Props {
  active: boolean;
  autoMs?: number | null;
  onFrame: (file: File, preview: string | null) => void;
}

export function CameraCapture({ active, autoMs = null, onFrame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onFrameRef = useRef(onFrame);
  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(false);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 } } })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play().catch(() => {});
        }
        setLive(true);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : 'Camera unavailable');
        setLive(false);
      });
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setLive(false);
    };
  }, [active]);

  const capture = useCallback((withPreview: boolean) => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
        onFrameRef.current(file, withPreview ? URL.createObjectURL(file) : null);
      },
      'image/jpeg',
      0.9
    );
  }, []);

  useEffect(() => {
    if (!live || !autoMs) return;
    const id = window.setInterval(() => capture(false), autoMs);
    return () => window.clearInterval(id);
  }, [live, autoMs, capture]);

  if (!active) return null;

  return (
    <div className="space-y-2">
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full rounded-lg bg-black max-h-80 object-contain ${autoMs ? '' : '-scale-x-100'}`}
        />
        {live && autoMs && (
          <span className="absolute top-2 right-2 text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/60 text-green-400">
            LIVE · sending 1 frame/{Math.round(autoMs / 1000)}s
          </span>
        )}
      </div>
      {error && <div className="text-red-400 text-sm">{error}</div>}
      {!error && !live && <div className="text-gray-500 text-sm">Starting camera...</div>}
      {live && !autoMs && (
        <button
          onClick={() => capture(true)}
          className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-sm text-gray-300 transition-colors"
        >
          Capture Frame
        </button>
      )}
    </div>
  );
}
