import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function MovieReviewChat() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchReview = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/query/?query=${encodeURIComponent(query)}`);
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse("Failed to fetch review.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <Card className="w-full max-w-lg p-4 shadow-md bg-white">
        <CardContent>
          <h1 className="text-xl font-bold text-center mb-4">Movie Review Q&A</h1>
          <div className="flex gap-2 mb-4">
            <Input 
              type="text" 
              placeholder="Ask about a movie..." 
              value={query} 
              onChange={(e) => setQuery(e.target.value)} 
              className="flex-1"
            />
            <Button onClick={fetchReview} disabled={loading}>
              {loading ? "Loading..." : "Ask"}
            </Button>
          </div>
          {response && (
            <div className="mt-4 p-3 border rounded bg-gray-50">
              <p className="text-sm text-gray-800">{response}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
