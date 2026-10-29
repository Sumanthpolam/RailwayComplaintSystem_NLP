const form = document.getElementById("complaint-form");
const textarea = document.getElementById("complaint");
const counter = document.getElementById("char-count");
const errorBox = document.getElementById("form-error");
const results = document.getElementById("results");

textarea.addEventListener("input", () => {
  counter.textContent = `${textarea.value.length} / 2000`;
  errorBox.hidden = true;
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    textarea.value = button.dataset.example;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  });
});

const departmentMap = {
  "AC/Electrical": ["Electrical Department", "Electrician"],
  "Food": ["Catering Department", "Catering Supervisor"],
  "Cleanliness": ["Housekeeping Department", "Housekeeping Staff"],
  "Toilet/Sanitation": ["Housekeeping Department", "Housekeeping Staff"],
  "Delay": ["Operations Department", "Operations Controller"],
  "Water": ["Water & Sanitation Department", "Water Maintenance Staff"],
  "Security": ["Railway Security", "Railway Security Officer"],
  "Staff/Service": ["Passenger Services", "Passenger Service Officer"],
  "Reservation/Seat": ["Reservation Department", "TTE / Reservation Officer"],
  "Luggage/Lost Property": ["Lost & Found Office", "Lost & Found Officer"],
  "Other": ["General Complaint Cell", "Complaint Cell Officer"]
};

function matches(text, patterns) {
  return patterns.some((pattern) => pattern.test(text));
}

function classifyComplaint(input) {
  const text = input.toLowerCase().replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim();

  // Check food first so food-quality words are not confused with generic cleanliness.
  if (matches(text, [
    /\b(rice|food|meal|catering|stale|spoiled|rotten|tasteless|unhygienic|contaminated|snack|breakfast|lunch|dinner)\b/,
    /food poisoning/, /poor quality food/, /served cold/, /food is cold/,
    /food contains/, /insects? in (the )?(food|meal|rice)/,
    /(food|meal|rice).{0,25}(bad|smell|dirty|quality|fresh|cold)/,
    /(bad|poor|stale|spoiled|rotten).{0,20}(food|meal|rice)/
  ])) return "Food";

  const rules = [
    ["Security", [/\b(unsafe|danger|threat|theft|stolen|steal|harass|security|fire|smoke|assault|molest|robbery|attack|emergency)\b/]],
    ["Toilet/Sanitation", [/\b(toilet|washroom|restroom|bathroom|flush|sanitation)\b/]],
    ["AC/Electrical", [/\b(ac|fan|light|electricity|electrical|socket|plug|charging point)\b/, /air condition/, /power failure/, /not cooling/]],
    ["Water", [/\bwater\b/, /water supply/, /water leakage/, /no drinking water/]],
    ["Cleanliness", [/\b(dirty|dust|garbage|unclean|waste|stain|cockroach|trash|cleaning|smell)\b/, /not clean/, /not been cleaned/]],
    ["Staff/Service", [/\b(staff|rude|disrespect|misbehav|attendant|unprofessional|porter)\b/, /refused to/, /did not respond/]],
    ["Delay", [/\b(delay|delayed|late|waiting|cancelled|canceled)\b/, /running late/]],
    ["Reservation/Seat", [/\b(seat|berth|reserved|reservation|pnr|ticket|coach allocation)\b/]],
    ["Luggage/Lost Property", [/\b(luggage|bag|suitcase|belongings|baggage)\b/, /lost property/, /lost my/]]
  ];
  for (const [category, patterns] of rules) {
    if (matches(text, patterns)) return category;
  }
  return "Other";
}

function getPriority(text, category) {
  const lower = text.toLowerCase();
  if (/\b(fire|smoke|theft|stolen|robbery|harassment|attack|fight|injury|accident|danger|unsafe|emergency|assault)\b/.test(lower)) return "Critical";
  if (/\b(delay|delayed|no water|water leakage|power failure|security)\b/.test(lower)) return "High";
  if (["AC/Electrical", "Food", "Water", "Staff/Service", "Security"].includes(category)) return "Medium";
  return "Low";
}

function createReference() {
  const now = new Date();
  const stamp = [
    String(now.getFullYear()).slice(-2),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0")
  ].join("");
  return `DEMO-${stamp}`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const complaint = textarea.value.trim();
  if (complaint.length < 5) {
    errorBox.textContent = "Please enter at least 5 characters describing the complaint.";
    errorBox.hidden = false;
    textarea.focus();
    return;
  }

  const category = classifyComplaint(complaint);
  const priority = getPriority(complaint, category);
  const [department, assignedTo] = departmentMap[category] || departmentMap.Other;

  document.getElementById("category").textContent = category;
  const priorityElement = document.getElementById("priority");
  priorityElement.className = `priority ${priority.toLowerCase()}`;
  priorityElement.querySelector("span").textContent = priority;
  document.getElementById("department").textContent = department;
  document.getElementById("assigned-to").textContent = assignedTo;
  document.getElementById("reference-id").textContent = createReference();
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
});

document.getElementById("new-complaint").addEventListener("click", () => {
  textarea.value = "";
  textarea.dispatchEvent(new Event("input"));
  results.hidden = true;
  textarea.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
