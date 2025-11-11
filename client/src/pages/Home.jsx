import { useEffect } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

function Badge({ children }) {
  return (
    <span className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-semibold text-amber-700">
      {children}
    </span>
  );
}

export default function Home() {
  // Ensure smooth scroll works even when arriving with /#hash from other pages
  useEffect(() => {
    if (window.location.hash) {
      const id = window.location.hash.slice(1);
      const el = document.getElementById(id);
      if (el) setTimeout(() => el.scrollIntoView({ behavior: "smooth" }), 50);
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-green-50">
      <Navbar />

      <main className="flex-1">
        {/* 1) HERO — full-bleed background, centered content */}
        <section id="hero" className="relative isolate" style={{ minHeight: "78vh" }}>
          <img
            src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=1600&auto=format&fit=crop"
            alt="Agricultural fields"
            className="absolute inset-0 h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-black/40" />
          <div className="relative z-10 mx-auto flex h-full max-w-6xl items-center justify-center px-6 text-center">
            <div className="max-w-2xl">
              <p className="mb-2 text-sm uppercase tracking-widest text-emerald-200/90">
                Smart Farming Assistance
              </p>
              <h1 className="text-4xl font-extrabold leading-tight text-white md:text-5xl">
                World’s most accessible <span className="text-emerald-300">AI advisor</span> for farmers
              </h1>
              <p className="mx-auto mt-3 max-w-xl text-emerald-100">
                AgroAware helps farmers pick the right crop, get fertilizer advice, and learn best practices—
                multilingual and field-ready.
              </p>

              <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                {/* Small blue button like reference site */}
                <a
                  href="#about"
                  className="rounded-full bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-700"
                >
                  Explore About Us
                </a>
                <a
                  href="/login"
                  className="rounded-full bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700"
                >
                  Try Advisory
                </a>
                <a
                  href="#videos"
                  className="rounded-full px-4 py-2 text-sm font-semibold text-emerald-200 hover:text-white underline"
                  title="Watch AI in Farming"
                >
                  Watch: AI in Farming
                </a>
              </div>
            </div>
          </div>
        </section>

        {/* 2) ABOUT */}
        <section id="about" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <div className="grid gap-10 md:grid-cols-2 md:items-center">
              <div>
                <h2 className="text-2xl font-bold text-green-800">About AgroAware</h2>
                <p className="mt-3 text-gray-700">
                  AgroAware is a Generative-AI farming advisor that helps farmers select crops,
                  get fertilizer guidance, and access easy learning materials. It also powers NGO
                  awareness drives with auto-generated multilingual posters and guides.
                </p>
                <ul className="mt-4 list-disc pl-5 text-gray-700">
                  <li>Crop & fertilizer recommendations (Expert mode)</li>
                  <li>District & season guidance (Beginner mode)</li>
                  <li>AI awareness content <Badge>Coming Soon</Badge></li>
                  <li>Scheme simplifier <Badge>Coming Soon</Badge></li>
                  <li>Voice assistant (KN/HI/TE) <Badge>Coming Soon</Badge></li>
                </ul>
              </div>

              {/* Updated About images (2 only) */}
              <div className="grid grid-cols-2 gap-3">
                <img
                  src="https://images.pexels.com/photos/175389/pexels-photo-175389.jpeg?auto=compress&cs=tinysrgb&w=1200"
                  alt="Farmer in field"
                  className="h-40 w-full rounded-xl object-cover"
                />
                <img
                  src="https://images.pexels.com/photos/236047/pexels-photo-236047.jpeg?auto=compress&cs=tinysrgb&w=1200"
                  alt="Soil testing"
                  className="h-40 w-full rounded-xl object-cover"
                />
              </div>
            </div>
          </div>
        </section>

        {/* 3) FEATURES */}
        <section id="features" className="bg-green-50">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-center text-2xl font-bold text-green-800">Key Features</h2>
            <div className="mt-8 grid gap-6 md:grid-cols-3">
              {[
                { icon: "🌾", title: "Smart Crop Advisory", desc: "Use N-P-K, pH, temperature & rainfall to get a crop + confidence." },
                { icon: "🧭", title: "Beginner Mode", desc: "No soil test? Select district & season to see suitable crops." },
                { icon: "🎨", title: <>Gen-AI Posters <Badge>Coming Soon</Badge></>, desc: "Instant awareness posters, slogans & tips." },
                { icon: "🏛", title: <>Scheme Simplifier <Badge>Coming Soon</Badge></>, desc: "Plain-language summaries of government schemes." },
                { icon: "🗣", title: <>Voice Assistant <Badge>Coming Soon</Badge></>, desc: "Ask in Kannada/Hindi/Telugu and hear answers." },
                { icon: "🌐", title: <>Multilingual UI <Badge>Coming Soon</Badge></>, desc: "Localized UI & content for rural outreach." },
              ].map((f) => (
                <div key={(typeof f.title === "string" ? f.title : "item") + f.icon} className="card">
                  <div className="text-3xl">{f.icon}</div>
                  <h3 className="mt-2 text-lg font-semibold text-green-800">{f.title}</h3>
                  <p className="mt-1 text-gray-700">{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* 4) HOW IT WORKS */}
        <section id="how" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-2xl font-bold text-green-800">How It Works</h2>
            <div className="mt-6 grid gap-6 md:grid-cols-4">
              {[
                { step: "1", title: "Create Account", text: "Sign up & choose your preferred language." },
                { step: "2", title: "Choose Mode", text: "Expert (soil values) or Beginner (district/season)." },
                { step: "3", title: "Get Advice", text: "See crop, fertilizer guidance & model confidence." },
                { step: "4", title: "Act & Learn", text: "Use awareness content & best practices." },
              ].map((s) => (
                <div key={s.step} className="rounded-2xl border bg-white p-5">
                  <div className="text-2xl font-bold text-green-700">Step {s.step}</div>
                  <div className="mt-1 text-lg font-semibold text-gray-800">{s.title}</div>
                  <p className="mt-1 text-gray-700">{s.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* 5) GALLERY (your 1, 4, 6 picks) */}
        <section id="gallery" className="bg-green-50">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-center text-2xl font-bold text-green-800">Field Gallery</h2>

            <div className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-3">
              <img
                src="https://images.pexels.com/photos/219794/pexels-photo-219794.jpeg?auto=compress&cs=tinysrgb&w=1200"
                alt="Field Cultivation"
                className="h-56 w-full rounded-xl object-cover shadow"
              />
              <img
                src="https://images.pexels.com/photos/2886937/pexels-photo-2886937.jpeg?auto=compress&cs=tinysrgb&w=1200"
                alt="Farmers at Work"
                className="h-56 w-full rounded-xl object-cover shadow"
              />
              <img
                src="https://images.pexels.com/photos/129574/pexels-photo-129574.jpeg?auto=compress&cs=tinysrgb&w=1200"
                alt="Crop Fields"
                className="h-56 w-full rounded-xl object-cover shadow"
              />
            </div>
          </div>
        </section>

        {/* 6) VIDEOS */}
        <section id="videos" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-center text-2xl font-bold text-green-800">Farmer Stories & AI in Agri</h2>
            <div className="mt-8 grid gap-6 md:grid-cols-2">
              <div className="aspect-video overflow-hidden rounded-xl shadow">
                <iframe
                  className="h-full w-full"
                  src="https://www.youtube.com/embed/2Vv-BfVoq04g"
                  title="AI in Agriculture - Explainer"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
              <div className="aspect-video overflow-hidden rounded-xl shadow">
                <iframe
                  className="h-full w-full"
                  src="https://www.youtube.com/embed/f77SKdyn-1Y"
                  title="Farmer Story"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            </div>
          </div>
        </section>

        {/* 7) FAQs */}
        <section id="faqs" className="bg-green-50">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-center text-2xl font-bold text-green-800">Frequently Asked Questions</h2>
            <div className="mt-8 grid gap-4 md:grid-cols-2">
              {[
                {
                  q: "Do I need a soil test to use AgroAware?",
                  a: "No. Use Beginner Mode with district & season. Expert Mode gives more precise results if you have soil values."
                },
                {
                  q: "Is AgroAware available in local languages?",
                  a: "Yes. Multilingual UI and voice assistant for Kannada, Hindi, and Telugu are planned in upcoming phases."
                },
                {
                  q: "Can NGOs create awareness posters?",
                  a: "Yes. The Generative-AI poster module will generate slogans, tips, and print-ready posters. Coming soon."
                },
                {
                  q: "What data do you store?",
                  a: "We store basic login info and advisory logs to show history/analytics. Data stays private to your account."
                },
              ].map(({q,a}) => (
                <details key={q} className="rounded-2xl border bg-white p-4">
                  <summary className="cursor-pointer font-semibold text-green-800">{q}</summary>
                  <p className="mt-2 text-gray-700">{a}</p>
                </details>
              ))}
            </div>
          </div>
        </section>

        {/* 8) CONTACT & SUPPORT */}
        <section id="contact" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 py-14">
            <h2 className="text-2xl font-bold text-green-800">Contact & Support</h2>
            <p className="mt-2 max-w-xl text-gray-700">
              Have questions, want a demo for your college review, or need deployment help?
              Send us a message.
            </p>
            <div className="mt-6 grid gap-6 md:grid-cols-2">
              <form
                className="card space-y-3"
                onSubmit={(e) => {
                  e.preventDefault();
                  const fd = new FormData(e.currentTarget);
                  const payload = Object.fromEntries(fd.entries());
                  console.log("Support form payload:", payload);
                  alert("Thanks! We’ll reach out soon.");
                  e.currentTarget.reset();
                }}
              >
                <input name="name" className="input" placeholder="Your Name" required />
                <input name="email" className="input" placeholder="Email / Phone" required />
                <textarea name="message" className="input h-32" placeholder="Your message..." required />
                <button className="btn w-full">Send Message</button>
              </form>

              <div className="card space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">📧</span>
                  <a className="text-green-700 underline" href="mailto:support@agroaware.example">
                    support@agroaware.example
                  </a>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">📞</span>
                  <span>+91 9945469518</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">📍</span>
                  <span>Department Lab,NMIT, Karnataka</span>
                </div>
                <div className="rounded-xl border bg-green-50 p-4 text-sm text-gray-700">
                  
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}