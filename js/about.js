// About page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const mobileMenu = document.getElementById('mobileMenu');
    const navLinks = document.querySelector('.nav-links');

    if (mobileMenu && navLinks) {
        mobileMenu.addEventListener('click', function() {
            navLinks.classList.toggle('mobile-open');
        });
    }

    // Smooth scrolling for internal links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated-element');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll(
        '.mission-content, .team-member, .tech-card, .stat, .partner-logo'
    );

    animatedElements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(element);
    });

    // Counter animation for statistics
    const counters = document.querySelectorAll('.stat .number');
    const animateCounter = (counter) => {
        const target = counter.textContent;
        const isPercentage = target.includes('%');
        const isCurrency = target.includes('₹') || target.includes('M') || target.includes('K');

        let numTarget;
        let suffix = '';

        if (isPercentage) {
            numTarget = parseInt(target.replace('%', ''));
            suffix = '%';
        } else if (target.includes('M')) {
            numTarget = parseFloat(target.replace(/[^\d.]/g, ''));
            suffix = 'M+';
        } else if (target.includes('K')) {
            numTarget = parseInt(target.replace(/[^\d]/g, ''));
            suffix = 'K';
        } else {
            numTarget = parseInt(target.replace(/[^\d]/g, ''));
        }

        let current = 0;
        const increment = numTarget / 50;
        const timer = setInterval(() => {
            current += increment;
            if (current >= numTarget) {
                current = numTarget;
                clearInterval(timer);
            }

            if (isCurrency && target.includes('₹')) {
                counter.textContent = `₹${Math.floor(current)}${suffix}`;
            } else if (suffix === 'M+') {
                counter.textContent = `${current.toFixed(1)}${suffix}`;
            } else {
                counter.textContent = `${Math.floor(current)}${suffix}`;
            }
        }, 20);
    };

    // Observe stats for counter animation
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                entry.target.classList.add('counted');
                const counter = entry.target.querySelector('.number');
                if (counter) {
                    animateCounter(counter);
                }
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.stat').forEach(stat => {
        statsObserver.observe(stat);
    });

    // Add hover effects to interactive elements
    const interactiveElements = document.querySelectorAll(
        '.team-member, .tech-card, .stat, .logo-placeholder'
    );

    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px)';
        });

        element.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Parallax effect for visual placeholder
    const visualPlaceholder = document.querySelector('.visual-placeholder');
    if (visualPlaceholder) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            visualPlaceholder.style.transform = `translateY(${rate}px)`;
        });
    }

    // Add loading animation
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease';
        document.body.style.opacity = '1';
    }, 100);

    // Easter egg - Konami code
    let konamiCode = [];
    const konamiSequence = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];

    document.addEventListener('keydown', (e) => {
        konamiCode.push(e.keyCode);
        if (konamiCode.length > konamiSequence.length) {
            konamiCode.shift();
        }

        if (JSON.stringify(konamiCode) === JSON.stringify(konamiSequence)) {
            showEasterEgg();
        }
    });

    function showEasterEgg() {
        const message = document.createElement('div');
        message.innerHTML = `
            <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        background: var(--primary-color); color: white; padding: 2rem;
                        border-radius: 1rem; z-index: 10000; text-align: center;
                        box-shadow: var(--shadow-heavy);">
                <h3>🎉 Easter Egg Found!</h3>
                <p>You discovered the Konami Code!</p>
                <p>Extra security level: <strong>ACTIVATED</strong> 🛡️</p>
                <button onclick="this.parentElement.parentElement.remove()"
                        style="margin-top: 1rem; padding: 0.5rem 1rem; border: none;
                               border-radius: 0.5rem; cursor: pointer;">Close</button>
            </div>
        `;
        document.body.appendChild(message);

        setTimeout(() => {
            if (message.parentNode) {
                message.remove();
            }
        }, 5000);
    }
});
