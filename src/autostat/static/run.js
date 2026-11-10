// The run panel of the autostat GUI: one button under the generated form
// sends its fields, as JSON, to the local host started by `autostat-gui`.
// Every visible string comes from locales/i18n.yaml through /i18n.json.
//
// Author: Warith Harchaoui <warithmetics@deraison.ai>

/**
 * Read the form's fields as the host expects them.
 *
 * @param {HTMLFormElement} form The generated form.
 * @returns {Object<string, (string|boolean)>} Field values by parameter name;
 *   empty text fields are left out so the command's defaults apply.
 */
function readFields(form) {
  const fields = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") {
      fields[element.name] = element.checked;
    } else if (element.value !== "") {
      fields[element.name] = element.value;
    }
  }
  return fields;
}

/**
 * Write the outcome of a run into the panel, as text nodes only.
 *
 * @param {HTMLElement} panel Where the result goes.
 * @param {Object} outcome The host's JSON answer.
 * @param {Object<string, string>} t The translated strings.
 * @returns {void}
 */
function showOutcome(panel, outcome, t) {
  panel.replaceChildren();
  const add = (tag, text) => {
    const node = document.createElement(tag);
    node.textContent = text;
    panel.append(node);
    return node;
  };
  add("h3", t["gui.run.structure"]).className = "font-semibold mt-2";
  add("p", `${outcome.structure}  (BIC ${outcome.bic.toFixed(1)})`).className = "font-mono";
  add("h3", t["gui.run.components"]).className = "font-semibold mt-2";
  const list = add("ol", "");
  list.className = "list-decimal ml-6";
  for (const summary of outcome.components) {
    const item = document.createElement("li");
    item.textContent = summary;
    list.append(item);
  }
  add("h3", t["gui.run.files"]).className = "font-semibold mt-2";
  const files = add("ul", "");
  for (const name of outcome.files) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `/runs/${outcome.run_id}/${encodeURI(name)}`;
    link.textContent = name;
    link.className = "underline";
    item.append(link);
    files.append(item);
  }
}

/**
 * Add the run button and its result panel under every generated form.
 *
 * @returns {Promise<void>} Resolves once the buttons are in place.
 */
async function install() {
  const lang = (navigator.language || "en").slice(0, 2);
  const t = await (await fetch(`/i18n.json?lang=${lang}`)).json();
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t[node.dataset.i18n] ?? "";
  }
  for (const form of document.querySelectorAll("form[data-cli-form]")) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = t["gui.run.button"];
    button.className =
      "ml-2 min-h-11 rounded-full bg-brand-blue px-5 py-2 font-semibold text-white " +
      "focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2";
    const panel = document.createElement("div");
    panel.setAttribute("role", "status");
    panel.setAttribute("aria-live", "polite");
    panel.className = "mt-3";
    form.append(button, panel);
    button.addEventListener("click", async () => {
      button.disabled = true;
      panel.textContent = t["gui.run.running"];
      try {
        const response = await fetch("/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(readFields(form)),
        });
        const body = await response.json();
        if (response.ok) {
          showOutcome(panel, body, t);
        } else {
          panel.textContent = `${t["gui.run.failed"]}: ${body.error.message}`;
        }
      } catch {
        panel.textContent = t["gui.run.network"];
      } finally {
        button.disabled = false;
      }
    });
  }
}

install();
