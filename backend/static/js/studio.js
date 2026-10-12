(() => {
  const input = document.getElementById("image-input");
  const zone = document.getElementById("upload-zone");
  const preview = document.getElementById("local-preview");
  const title = document.getElementById("upload-title");
  const selected = document.getElementById("selected-file");
  if (input) input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) { alert("Please choose an image file."); input.value = ""; return; }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    title.textContent = "Image selected";
    selected.textContent = file.name;
  });
  if (zone) {
    ["dragenter","dragover"].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("dragging"); }));
    ["dragleave","drop"].forEach(evt => zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove("dragging"); }));
    zone.addEventListener("drop", e => {
      const file = e.dataTransfer.files[0];
      if (!file) return;
      const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
      input.dispatchEvent(new Event("change"));
    });
  }

  const video = document.getElementById("camera-video");
  if (!video) return;
  const canvas = document.getElementById("camera-canvas");
  const placeholder = document.getElementById("camera-placeholder");
  const startBtn = document.getElementById("start-camera");
  const stopBtn = document.getElementById("stop-camera");
  const captureBtn = document.getElementById("capture-frame");
  const status = document.getElementById("camera-status");
  const liveResults = document.getElementById("live-results");
  let stream = null;
  let busy = false;

  function stopCamera() {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    video.classList.remove("active");
    placeholder.style.display = "flex";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    captureBtn.disabled = true;
    status.textContent = "Camera is off. Start camera to enable live capture.";
  }
  startBtn.addEventListener("click", async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status.textContent = "Camera access is unavailable in this browser/context. Use localhost or HTTPS.";
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}, audio: false});
      video.srcObject = stream;
      await video.play();
      video.classList.add("active");
      placeholder.style.display = "none";
      startBtn.disabled = true;
      stopBtn.disabled = false;
      captureBtn.disabled = false;
      status.textContent = "Camera active. Capture a frame to run segmentation.";
    } catch (err) {
      status.textContent = "Could not start camera: " + err.message;
    }
  });
  stopBtn.addEventListener("click", stopCamera);

  captureBtn.addEventListener("click", async () => {
    if (!stream || busy || !video.videoWidth) return;
    busy = true; captureBtn.disabled = true; status.textContent = "Running model inference…";
    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
      const form = new FormData(); form.append("image", blob, "camera-frame.jpg");
      const response = await fetch("/api/predict", {method: "POST", body: form});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed");
      document.getElementById("live-original").src = data.original;
      document.getElementById("live-mask").src = data.mask;
      document.getElementById("live-overlay").src = data.overlay;
      document.getElementById("download-live-original").href = data.original;
      document.getElementById("download-live-mask").href = data.mask;
      document.getElementById("download-live-overlay").href = data.overlay;
      document.getElementById("live-stats").innerHTML = '<div class="panel-heading"><span class="step">CLASS DISTRIBUTION</span><span class="tiny-badge">' + data.width + ' × ' + data.height + '</span></div>' +
        data.stats.map(item => '<div class="distribution-row"><span class="legend-dot" style="background:'+item.color+'"></span><span class="distribution-name">'+item.name+'</span><div class="distribution-track"><i style="width:'+item.percent+'%;background:'+item.color+'"></i></div><strong>'+item.percent+'%</strong></div>').join("");
      liveResults.hidden = false;
      status.textContent = "Prediction complete. Capture another frame whenever ready.";
      liveResults.scrollIntoView({behavior: "smooth", block: "nearest"});
    } catch (err) {
      status.textContent = "Prediction error: " + err.message;
    } finally {
      busy = false; captureBtn.disabled = !stream;
    }
  });
  window.addEventListener("beforeunload", stopCamera);
})();
