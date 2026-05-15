document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.metric-bar-fill').forEach(bar => {
    const w = bar.style.width;
    bar.style.width = '0%';
    setTimeout(() => bar.style.width = w, 200);
  });
});