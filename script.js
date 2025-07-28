function copyBibTeX() {
    var bibTexElement = document.querySelector(".bibtex-section pre code");
    var bibTexText = bibTexElement.innerText.trim();
    navigator.clipboard.writeText(bibTexText);

    var message = document.getElementById("copyMessage");
    message.style.opacity = 1;

    // hide message after 5 seconds
    setTimeout(() => {
      message.style.opacity = 0;
    }, 5000);
  }
  function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    document.querySelector(".nav").classList.toggle("dark-mode");
  }
  window.onscroll = function () {
    const scrollUpBtn = document.getElementById("scrollUpBtn");
    if (
      document.body.scrollTop > 100 ||
      document.documentElement.scrollTop > 100
    ) {
      scrollUpBtn.style.display = "block";
    } else {
      scrollUpBtn.style.display = "none";
    }
  };
  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  class Carousel {
    constructor(element, interval = 3000) {
      this.container = element;
      this.track = element.querySelector(".carousel-track");
      this.slides = Array.from(element.querySelectorAll(".carousel-slide"));
      this.indicators = element.querySelector(".carousel-indicators");

      this.currentIndex = 0;
      this.slidesPerView = 1;
      this.totalSlides = this.slides.length;
      this.interval = interval;
      this.autoPlayTimer = null;

      this.createIndicators();
      this.setupEventListeners();
      this.startAutoPlay();
      this.updateCarousel();
    }

    createIndicators() {
      for (let i = 0; i < this.totalSlides; i++) {
        const button = document.createElement("button");
        button.classList.add("indicator");
        if (i === 0) button.classList.add("active");
        button.addEventListener("click", () => {
          this.goToSlide(i);
        });
        this.indicators.appendChild(button);
      }
    }

    setupEventListeners() {
      this.container
        .querySelector(".prev")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.prevSlide();
        });

      this.container
        .querySelector(".next")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.nextSlide();
        });

      this.container.addEventListener("mouseenter", () => {
        this.stopAutoPlay();
      });

      this.container.addEventListener("mouseleave", () => {
        this.startAutoPlay();
      });

      document.addEventListener("keydown", (e) => {
        if (this.container.matches(":hover")) {
          if (e.key === "ArrowLeft") {
            this.prevSlide();
          } else if (e.key === "ArrowRight") {
            this.nextSlide();
          }
        }
      });
    }

    updateCarousel() {
      const offset = -this.currentIndex * 100;
      this.track.style.transform = `translateX(${offset}%)`;

      const indicators = Array.from(this.indicators.children);
      indicators.forEach((indicator, index) => {
        indicator.classList.toggle("active", index === this.currentIndex);
      });
    }

    nextSlide() {
      this.currentIndex = (this.currentIndex + 1) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    prevSlide() {
      this.currentIndex =
        (this.currentIndex - 1 + this.totalSlides) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    goToSlide(index) {
      if (index !== this.currentIndex) {
        this.currentIndex = index;
        this.updateCarousel();
        this.resetAutoPlay();
      }
    }

    startAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
      }
      this.autoPlayTimer = setInterval(() => {
        this.nextSlide();
      }, this.interval);
    }

    stopAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
      }
    }

    resetAutoPlay() {
      this.stopAutoPlay();
      this.startAutoPlay();
    }
  }

  function closeModal() {
    document.getElementById("licenseModal").style.display = "none";
  }

  document.addEventListener("DOMContentLoaded", () => {
    // begin modal code for license
    const checkpointsUrl = "https://datashare.tu-dresden.de/s/Ck6AKFjka6StgCw";
    const checkpointLink = document.getElementById("checkpoints-link");
    if (checkpointLink) {
      checkpointLink.style.display = "inline-block";
    }
      // show real link if license already accepted
    if (sessionStorage.getItem("checkpoints_license_accepted")) {
      checkpointLink.href = checkpointsUrl;
    }
    const modal = document.getElementById("licenseModal");
    const licenseText = document.getElementById("licenseText");
    const acceptBtn = document.getElementById("acceptBtn");

    // show modal on checkpoints click if license not yet accepted
    if (checkpointLink) {
      checkpointLink.addEventListener("click", (e) => {
        if (!sessionStorage.getItem("checkpoints_license_accepted")) {
          e.preventDefault();
          modal.style.display = "block";
        }
      });
    }

    // enable accept button after scrolling to bottom
    licenseText.addEventListener("scroll", () => {
      if (
        licenseText.scrollTop + licenseText.clientHeight >=
        licenseText.scrollHeight
      ) {
        acceptBtn.disabled = false;
      }
    });

    // Accept license and open link
    acceptBtn.addEventListener("click", () => {
      sessionStorage.setItem("checkpoints_license_accepted", "true");
      checkpointLink.href = checkpointsUrl;
      modal.style.display = "none";
      window.open(checkpointsUrl, "_blank");
      return false;
    });
    // end modal code
    const imageCarouselEl = document.querySelector("#imageCarousel");
    const videoCarouselEl = document.querySelector("#videoCarousel");
    const carousels = [];
    if (imageCarouselEl) {
      carousels.push(new Carousel(imageCarouselEl, 10000));
    }
    if (videoCarouselEl) {
      carousels.push(new Carousel(videoCarouselEl, 5000));
    }

    // Add touch support
    carousels.forEach((carousel) => {
      let touchStartX = 0;
      let touchEndX = 0;

      carousel.container.addEventListener(
        "touchstart",
        (e) => {
          touchStartX = e.changedTouches[0].screenX;
        },
        { passive: true }
      );

      carousel.container.addEventListener(
        "touchend",
        (e) => {
          touchEndX = e.changedTouches[0].screenX;
          handleSwipe(carousel);
        },
        { passive: true }
      );

      function handleSwipe(carousel) {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
          if (diff > 0) {
            carousel.nextSlide();
          } else {
            carousel.prevSlide();
          }
        }
      }
    });
  });