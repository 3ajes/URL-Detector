import React, { useState } from 'react';
import { Shield, Search, AlertTriangle, CheckCircle, Loader, Link as LinkIcon } from 'lucide-react';

export default function App() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      // After
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url: url })
});

      if (!response.ok) throw new Error('API Error');

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Error connecting to ML model. Is the backend running?');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence < 30) return 'text-green-600';
    if (confidence < 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBg = (confidence) => {
    if (confidence < 30) return 'bg-green-100';
    if (confidence < 70) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">PhishGuard</h1>
              <p className="text-gray-600">AI-powered Phishing URL Detection</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Protect Yourself from Phishing Attacks with Esra phishing Detection
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Enter any URL to instantly analyze it for phishing threats using our advanced ML model
          </p>
        </div>

       
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-200">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <LinkIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com or example.com"
                className="w-full pl-12 pr-4 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all duration-200"
              />
            </div>
            
            {error && (
              <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
                <AlertTriangle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader className="h-5 w-5 animate-spin" />
                  <span>Analyzing URL...</span>
                </>
              ) : (
                <>
                  <Search className="h-5 w-5" />
                  <span>Scan for Phishing</span>
                </>
              )}
            </button>
          </form>
        </div>

        {result && (
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
            <div className="text-center mb-6">
              <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${
                result.isPhishing 
                  ? 'bg-red-100 text-red-600' 
                  : 'bg-green-100 text-green-600'
              }`}>
                {result.isPhishing ? (
                  <AlertTriangle className="h-8 w-8" />
                ) : (
                  <CheckCircle className="h-8 w-8" />
                )}
              </div>
              
              <h3 className={`text-2xl font-bold mb-2 ${
                result.isPhishing ? 'text-red-600' : 'text-green-600'
              }`}>
                {result.isPhishing ? 'Phishing Detected!' : 'URL Appears Safe'}
              </h3>
              
              <p className={`text-lg font-medium ${getConfidenceColor(result.confidence)}`}>
                Confidence: {result.confidence.toFixed(1)}%
              </p>
            </div>

    
            <div className="mb-6">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Low Risk</span>
                <span>High Risk</span>
              </div>
              <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-1000 ${
                    getConfidenceBg(result.confidence)
                  }`}
                  style={{ width: `${result.confidence}%` }}
                />
              </div>
            </div>

          
            <div className="bg-gray-50 rounded-xl p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Recommendations:</h4>
              <ul className="space-y-2">
                {result.recommendations?.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className={`w-2 h-2 rounded-full mt-2 ${
                      result.isPhishing ? 'bg-red-500' : 'bg-green-500'
                    }`} />
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-6 text-sm text-gray-500 text-center">
              <p>Analysis completed in {result.analysisTime?.toFixed(2)} seconds</p>
              <p className="mt-1">Threat type: {result.threatType}</p>
            </div>
          </div>
        )}
      </main>

      
      <footer className="bg-gray-900 text-white py-8 mt-16">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-gray-400">
            © 2025 PhishGuard. This tool uses advanced ML algorithms with 97% Accuracy to help identify potential phishing threats. 
            Always exercise caution when clicking links.
          </p>
        </div>
      </footer>
    </div>
  );
}
