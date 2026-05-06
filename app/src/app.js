const form = document.getElementById("chat-form");
const queryEl = document.getElementById("query");
const personaEl = document.getElementById("persona-speaker");
const resultEl = document.getElementById("result");

function render(data) {
  const topics = (data.used_topics || []).join(", ") || "None";
  const chunks = (data.used_chunks || []).join(", ") || "None";
  const checkpoints = (data.used_checkpoints || []).join(", ") || "None";
  const facts = (data.used_facts || []).join(", ") || "None";

  const card = document.createElement("article");
  card.className = "card";

  const title = document.createElement("h2");
  title.textContent = "Answer";

  const pre = document.createElement("pre");
  pre.textContent = data.answer || "No answer returned.";

  const topicsLine = document.createElement("p");
  const topicsLabel = document.createElement("strong");
  topicsLabel.textContent = "Topics:";
  topicsLine.append(topicsLabel, ` ${topics}`);

  const chunksLine = document.createElement("p");
  const chunksLabel = document.createElement("strong");
  chunksLabel.textContent = "Chunks:";
  chunksLine.append(chunksLabel, ` ${chunks}`);

  const checkpointsLine = document.createElement("p");
  const checkpointsLabel = document.createElement("strong");
  checkpointsLabel.textContent = "Checkpoints:";
  checkpointsLine.append(checkpointsLabel, ` ${checkpoints}`);

  const factsLine = document.createElement("p");
  const factsLabel = document.createElement("strong");
  factsLabel.textContent = "Facts:";
  factsLine.append(factsLabel, ` ${facts}`);

  card.append(title, pre, topicsLine, chunksLine, checkpointsLine, factsLine);
  resultEl.replaceChildren(card);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryEl.value.trim();
  if (!query) return;
  const thinkingCard = document.createElement("article");
  thinkingCard.className = "card";
  const thinkingText = document.createElement("p");
  thinkingText.textContent = "Thinking...";
  thinkingCard.append(thinkingText);
  resultEl.replaceChildren(thinkingCard);
  try {
    const res = await fetch("/.netlify/functions/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, persona_speaker: personaEl?.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    render(data);
  } catch (err) {
    const errorCard = document.createElement("article");
    errorCard.className = "card error";
    const errorText = document.createElement("p");
    errorText.textContent = err.message;
    errorCard.append(errorText);
    resultEl.replaceChildren(errorCard);
  }
});

queryEl.value = "What kind of person is this user?";

