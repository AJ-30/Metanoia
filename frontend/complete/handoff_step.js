// Added to the plan-completion page (Reconstruction/serve/static/index.html) by frontend/completion.py.
//
// After a plan has been completed, offer the two furniture pages of this demo. The server converts the completed plan
// to the two class-id masks those pages read and stores them under a short id; the page that opens (/ or /inpaint)
// fetches them from /handoff/<id>/ and fills its file inputs. Uses the page's own helpers ($, api, esc, PROFILE).
window.NEXT_STEP = function (token) {
  const box = $("next-step");
  box.innerHTML = `<h3 style="font-size:1rem">Next: furniture</h3><p class="hint">Send the completed plan to the furniture models of this demo.</p>` +
    `<div class="row"><button type="button" class="go" id="h-upload">Place one piece (TV, showcase or dining set)</button>` +
    `<button type="button" class="go" id="h-inpaint">Edit furniture and fill all three</button></div><div id="hout"></div>`;
  const send = async (target) => {
    $("h-upload").disabled = $("h-inpaint").disabled = true;
    $("hout").innerHTML = `<p class="hint"><span class="spin"></span>Converting the plan…</p>`;
    try {
      const h = await api("/handoff", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: PROFILE, composed_token: token }) });
      window.location.href = `${target}?plan=${encodeURIComponent(h.id)}`;
    } catch (e) {
      $("hout").innerHTML = `<div class="msg err">${esc(e.message)}</div>`;
      $("h-upload").disabled = $("h-inpaint").disabled = false;
    }
  };
  $("h-upload").addEventListener("click", () => send("/"));
  $("h-inpaint").addEventListener("click", () => send("/inpaint"));
};
