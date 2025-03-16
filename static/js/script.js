const translations = [
    {
        "reviewer": "Ashwin M.",
        "review": {
            "original": "Value for money. A quality product, soft, good-looking.",
            "bn": "মূল্যের জন্য সেরা। একটি মানসম্পন্ন পণ্য, নরম এবং দেখতে সুন্দর।",
            "mr": "किंमतीसाठी उत्तम. दर्जेदार उत्पादन, मऊ आणि सुंदर दिसणारे."
        }
    },
    {
        "reviewer": "Ravi S.",
        "review": {
            "original": "Good quality fluffy teddy bear. Ordered a teddy bear for my mom during the Diwali sale from Jam & Honey...",
            "bn": "ভালো মানের ফ্লাফি টেডি বিয়ার। দীপাবলির বিক্রিতে মায়ের জন্য জ্যাম অ্যান্ড হানি থেকে একটি টেডি বিয়ার অর্ডার করেছিলাম...",
            "mr": "उत्तम दर्जाचा मऊ टेडी बेअर. दिवाळीच्या सेलमध्ये आईसाठी जॅम अँड हनी मधून टेडी बेअर ऑर्डर केला..."
        }
    },
    {
        "reviewer": "Pooja R.",
        "review": {
            "original": "Adorable and Perfect Gift for Kids. The Amazon Brand - Jam & Honey Teddy Bear is absolutely delightful!...",
            "bn": "আকর্ষণীয় এবং শিশুদের জন্য নিখুঁত উপহার। অ্যামাজন ব্র্যান্ড - জ্যাম অ্যান্ড হানি টেডি বিয়ার একেবারে আনন্দদায়ক!...",
            "mr": "गोंडस आणि मुलांसाठी परिपूर्ण भेट. अॅमेझॉन ब्रँड - जॅम अँड हनी टेडी बेअर एकदम अप्रतिम आहे!..."
        }
    }
];

function changeText(elementId, newText) {
    let element = document.getElementById(elementId);
    if (element) {
        element.innerText = newText;
    } else {
        console.error("Element with ID " + elementId + " not found.");
    }
}

function translateText(button) {
    let reviewContainer = button.closest(".review");
    let textElement = reviewContainer.querySelector("p[id]");
    let languageSelector = reviewContainer.querySelector("select");
    
    if (textElement && languageSelector) {
        let selectedLanguage = languageSelector.value;
        let originalText = textElement.innerText;
        
        let translations = {
            "bn": "এই পর্যালোচনাটি বাংলায় অনুবাদ করা হয়েছে।",
            "mr": "हे पुनरावलोकन मराठीत भाषांतरित केले आहे."
        };
        
        textElement.innerText = translations[selectedLanguage] || "Translation not available";
    }
}