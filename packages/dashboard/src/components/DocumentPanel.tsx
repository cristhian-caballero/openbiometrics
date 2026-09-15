import { useState } from 'react';
import { scanDocument, verifyDocument, type DocumentScanResponse, type VerifyResponse } from '../api';
import { CameraCapture } from './CameraCapture';
import { ImageDropZone } from './ImageDropZone';

type SelfieSource = 'upload' | 'camera';

export function DocumentPanel() {
  const [docFile, setDocFile] = useState<File | null>(null);
  const [docPreview, setDocPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DocumentScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Verify against selfie
  const [selfieSource, setSelfieSource] = useState<SelfieSource>('upload');
  const [selfieFile, setSelfieFile] = useState<File | null>(null);
  const [selfiePreview, setSelfiePreview] = useState<string | null>(null);
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null);
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  const handleCameraFrame = (file: File, preview: string | null) => {
    setSelfieFile(file);
    setSelfiePreview(preview);
    setVerifyResult(null);
  };

  const handleScan = async () => {
    if (!docFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setVerifyResult(null);

    try {
      const res = await scanDocument(docFile);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Document scan failed');
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async () => {
    if (!docFile || !selfieFile) return;
    setVerifyLoading(true);
    setVerifyError(null);
    setVerifyResult(null);

    try {
      const res = await verifyDocument(docFile, selfieFile);
      setVerifyResult(res);
    } catch (e) {
      setVerifyError(e instanceof Error ? e.message : 'Verification failed');
    } finally {
      setVerifyLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold mb-2">Document Scanner</h2>
        <p className="text-gray-400 text-sm">
          Upload a document image to extract MRZ data, OCR text, and detect faces.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <ImageDropZone
            onImage={(f, p) => {
              setDocFile(f);
              setDocPreview(p);
              setResult(null);
              setVerifyResult(null);
              setSelfieSource('upload');
            }}
            preview={docPreview}
            label="Drop document image here"
          />

          <button
            onClick={handleScan}
            disabled={!docFile || loading}
            className="w-full px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors"
          >
            {loading ? 'Scanning...' : 'Scan Document'}
          </button>

          {loading && (
            <div className="flex items-center gap-3 text-indigo-400 text-sm">
              <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              Processing document...
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        <div className="space-y-4">
          {result && (
            <>
              {/* Document type badge */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-4">
                <div className="flex items-center gap-3">
                  <span className="px-3 py-1 bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 rounded-full text-sm font-medium">
                    {result.document_type || 'Unknown'}
                  </span>
                  {result.mrz?.valid && (
                    <span className="px-3 py-1 bg-green-600/20 text-green-400 border border-green-500/30 rounded-full text-xs font-medium">
                      MRZ Valid
                    </span>
                  )}
                  {result.mrz && !result.mrz.valid && (
                    <span className="px-3 py-1 bg-red-600/20 text-red-400 border border-red-500/30 rounded-full text-xs font-medium">
                      MRZ Invalid
                    </span>
                  )}
                  {result.face && (
                    <span className="px-3 py-1 bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 rounded-full text-xs font-medium">
                      Face detected
                    </span>
                  )}
                </div>

                {/* MRZ parsed fields */}
                {result.mrz && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-400">MRZ Data</h3>
                    <div className="grid grid-cols-2 gap-3">
                      <MRZField label="Surname" value={result.mrz.parsed.surname} />
                      <MRZField label="Given Names" value={result.mrz.parsed.given_names} />
                      <MRZField label="Date of Birth" value={result.mrz.parsed.date_of_birth} />
                      <MRZField label="Nationality" value={result.mrz.parsed.nationality} />
                      <MRZField label="Document No." value={result.mrz.parsed.document_number} />
                      <MRZField label="Expiry Date" value={result.mrz.parsed.expiry_date} />
                      <MRZField label="Sex" value={result.mrz.parsed.sex} />
                      <MRZField label="Country" value={result.mrz.parsed.country} />
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <div className={`w-2 h-2 rounded-full ${result.mrz.check_digits_ok ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className={result.mrz.check_digits_ok ? 'text-green-400' : 'text-red-400'}>
                        Check digits {result.mrz.check_digits_ok ? 'valid' : 'invalid'}
                      </span>
                    </div>
                  </div>
                )}

                {/* OCR text */}
                {result.ocr && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-gray-400">OCR Text</h3>
                    <pre className="bg-gray-800/50 rounded-lg p-3 text-xs text-gray-300 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
                      {result.ocr.full_text}
                    </pre>
                  </div>
                )}
              </div>

              {/* Verify against selfie */}
              <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-gray-400">Verify Against Selfie</h3>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setSelfieSource('upload')}
                      className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                        selfieSource === 'upload'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      Upload
                    </button>
                    <button
                      onClick={() => setSelfieSource('camera')}
                      className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                        selfieSource === 'camera'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                      }`}
                    >
                      Camera
                    </button>
                  </div>
                </div>

                {selfieSource === 'upload' ? (
                  <ImageDropZone
                    onImage={(f, p) => {
                      setSelfieFile(f);
                      setSelfiePreview(p);
                      setVerifyResult(null);
                    }}
                    preview={selfiePreview}
                    label="Drop selfie image here"
                  />
                ) : (
                  <CameraCapture active={true} onFrame={handleCameraFrame} />
                )}
                <button
                  onClick={handleVerify}
                  disabled={!selfieFile || verifyLoading}
                  className="w-full px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors"
                >
                  {verifyLoading ? 'Verifying...' : 'Compare to Document Face'}
                </button>

                {verifyError && (
                  <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
                    {verifyError}
                  </div>
                )}

                {verifyResult && (
                  <>
                    <div className={`rounded-lg border p-4 text-center space-y-2 ${
                      verifyResult.is_match
                        ? 'bg-green-500/5 border-green-500/30'
                        : 'bg-red-500/5 border-red-500/30'
                    }`}>
                      <div className={`text-2xl font-bold ${verifyResult.is_match ? 'text-green-400' : 'text-red-400'}`}>
                        {verifyResult.is_match ? 'MATCH' : 'NO MATCH'}
                      </div>
                      <div className="text-sm">
                        Similarity: <span className="font-mono font-bold">{(verifyResult.similarity * 100).toFixed(2)}%</span>
                      </div>
                    </div>

                    {verifyResult.face2?.liveness && (
                      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
                        <h3 className="text-sm font-medium text-gray-400">Selfie Liveness Check</h3>
                        {verifyResult.face2.liveness.is_live === null ? (
                          <p className="text-xs text-gray-500">Liveness model not available</p>
                        ) : (
                          <div className="flex items-center justify-between">
                            <span className={`px-3 py-1 rounded-full text-xs font-medium border ${
                              verifyResult.face2.liveness.is_live
                                ? 'bg-green-600/20 text-green-400 border-green-500/30'
                                : 'bg-red-600/20 text-red-400 border-red-500/30'
                            }`}>
                              {verifyResult.face2.liveness.is_live ? 'LIVE' : 'SPOOF'}
                            </span>
                            {verifyResult.face2.liveness.score !== null && (
                              <span className="text-sm text-gray-300">
                                Score: <span className="font-mono font-bold">
                                  {(verifyResult.face2.liveness.score * 100).toFixed(1)}%
                                </span>
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MRZField({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-2.5">
      <div className="text-gray-500 text-xs">{label}</div>
      <div className="font-medium text-sm">{value || '-'}</div>
    </div>
  );
}
