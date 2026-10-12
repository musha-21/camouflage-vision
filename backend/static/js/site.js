const menuToggle = document.getElementById("menu-toggle");
const navLinks = document.getElementById("nav-links");
if (menuToggle && navLinks) {
  menuToggle.addEventListener("click", () => navLinks.classList.toggle("open"));
}

// Subtle pointer-driven 3D tilt on desktop.
document.querySelectorAll(".tilt-card").forEach(card => {
  card.addEventListener("pointermove", event => {
    if (window.innerWidth < 800) return;
    const rect = card.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    card.style.transform = `perspective(900px) rotateX(${(0.5-y)*5}deg) rotateY(${(x-0.5)*6}deg) translateY(-2px)`;
  });
  card.addEventListener("pointerleave", () => card.style.transform = "");
});
