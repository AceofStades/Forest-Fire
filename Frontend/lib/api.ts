const API_BASE = "http://localhost:8000";

export const fetchFireGrid = async () => {
  const response = await fetch(`${API_BASE}/fire-grid`);
  const data = await response.json();
  return data.grid; // Returns the 320x400 matrix
};

export const fetchSafePath = async (start: [number, number], goal: [number, number]) => {
  const response = await fetch(`${API_BASE}/get-safe-path`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start, goal }),
  });
  const data = await response.json();
  return data.path; // Returns [[r1, c1], [r2, c2]...]
};