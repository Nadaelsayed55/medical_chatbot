const API_BASE = "http://127.0.0.1:8000";

let currentPatientContext = null;
let chatHistory = [];

const $ = (id) => document.getElementById(id);

function getNumber(id) {
    return Number($(id).value);
}

function bmiCategory(bmi) {
    if (bmi < 18.5) return "نقص في الوزن";
    if (bmi < 25) return "وزن طبيعي";
    if (bmi < 30) return "زيادة في الوزن";
    if (bmi < 35) return "سمنة من الدرجة الأولى";
    if (bmi < 40) return "سمنة من الدرجة الثانية";
    return "سمنة شديدة";
}

function calculateBMI() {
    const height = getNumber("height");
    const weight = getNumber("weight");

    if (!height || !weight || height <= 0 || weight <= 0) {
        $("bmiPreview").textContent = "سيتم حسابه تلقائيًا";
        $("bmiCategory").textContent = "أدخل الطول والوزن أولًا";
        return null;
    }

    const bmi = weight / Math.pow(height / 100, 2);
    const rounded = Number(bmi.toFixed(1));

    $("bmiPreview").textContent = rounded;
    $("bmiCategory").textContent = bmiCategory(bmi);

    return rounded;
}

$("height").addEventListener("input", calculateBMI);
$("weight").addEventListener("input", calculateBMI);

function showError(message) {
    $("formError").textContent = message;
    $("formError").classList.remove("hidden");
}

function clearError() {
    $("formError").classList.add("hidden");
    $("formError").textContent = "";
}

function getRequiredValue(id, label) {
    const value = $(id).value;
    if (value === "") {
        throw new Error(`من فضلك اختر: ${label}`);
    }
    return Number(value);
}

function collectPatientData() {
    const height = getNumber("height");
    const weight = getNumber("weight");
    const bmi = calculateBMI();

    if (!height || height < 100 || height > 250) {
        throw new Error("أدخل طولًا صحيحًا بين 100 و250 سم.");
    }

    if (!weight || weight < 20 || weight > 300) {
        throw new Error("أدخل وزنًا صحيحًا بين 20 و300 كجم.");
    }

    if (!bmi) {
        throw new Error("تعذر حساب BMI.");
    }

    return {
        height_cm: height,
        weight_kg: weight,
        age_category: getRequiredValue("ageCategory", "الفئة العمرية"),
        gen_hlth: getRequiredValue("genHlth", "الصحة العامة"),
        high_bp: getRequiredValue("highBp", "ضغط الدم"),
        high_chol: getRequiredValue("highChol", "الكوليسترول"),
        smoker: getRequiredValue("smoker", "التدخين"),
        stroke: getRequiredValue("stroke", "السكتة الدماغية"),
        heart_disease: getRequiredValue("heartDisease", "أمراض القلب"),
        phys_activity: getRequiredValue("physActivity", "النشاط البدني"),
        diff_walk: getRequiredValue("diffWalk", "صعوبة المشي أو الحركة")
    };
}

function formatAssistantText(text) {
    // Keep line breaks and make simple markdown-style bold readable.
    return text
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .trim();
}

function resetChat() {
    chatHistory = [];
    $("chatMessages").innerHTML = `
        <div class="message assistant">
            <div class="avatar">🤖</div>
            <div class="bubble">
                أنا جاهز. اسألني عن نتيجة التحليل أو أي شيء متعلق بالبيانات الصحية التي أدخلتها.
            </div>
        </div>
    `;
}

function addMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);

    $("chatMessages").appendChild(wrapper);
    $("chatMessages").scrollTop = $("chatMessages").scrollHeight;
}

async function analyzePatient() {
    clearError();

    let data;
    try {
        data = collectPatientData();
    } catch (error) {
        showError(error.message);
        return;
    }

    $("analyzeBtn").disabled = true;
    $("analyzeBtn").textContent = "جاري التحليل...";

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "حدث خطأ أثناء التحليل.");
        }

        // This is the key part:
        // Save the exact patient context returned by the backend.
        // The chatbot will use this same context for every question.
        currentPatientContext = result.patient_context;

        $("resultBmi").textContent = result.bmi;
        $("resultBmiCategory").textContent = result.bmi_category;
        $("resultAge").textContent = result.age_label;
        $("resultStatus").textContent = result.status;
        $("aiAdvice").textContent = formatAssistantText(result.ai_advice);

        $("resultSection").classList.remove("hidden");
        $("chatSection").classList.remove("hidden");

        resetChat();

        // Scroll smoothly to the result.
        $("resultSection").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        showError(
            `${error.message} — تأكدي أن FastAPI يعمل على ${API_BASE}`
        );
    } finally {
        $("analyzeBtn").disabled = false;
        $("analyzeBtn").textContent = "تحليل البيانات وإظهار النتيجة 🚀";
    }
}

async function sendChatMessage(customMessage = null) {
    if (!currentPatientContext) {
        $("chatError").textContent = "اعملي تحليل للبيانات أولًا حتى يكون الشات مرتبطًا ببيانات المريض.";
        $("chatError").classList.remove("hidden");
        return;
    }

    $("chatError").classList.add("hidden");

    const message = (customMessage || $("chatInput").value).trim();

    if (!message) return;

    addMessage("user", message);
    $("chatInput").value = "";

    chatHistory.push({
        role: "user",
        content: message
    });

    $("sendChatBtn").disabled = true;
    $("sendChatBtn").textContent = "جاري الرد...";

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message,
                patient_context: currentPatientContext,
                conversation: chatHistory.slice(-12)
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "حدث خطأ في الشات.");
        }

        const answer = formatAssistantText(result.answer);

        addMessage("assistant", answer);

        chatHistory.push({
            role: "assistant",
            content: answer
        });

    } catch (error) {
        $("chatError").textContent = error.message;
        $("chatError").classList.remove("hidden");

        // Keep history clean if request failed.
        chatHistory.pop();
    } finally {
        $("sendChatBtn").disabled = false;
        $("sendChatBtn").textContent = "إرسال";
        $("chatInput").focus();
    }
}

$("analyzeBtn").addEventListener("click", analyzePatient);

$("sendChatBtn").addEventListener("click", () => {
    sendChatMessage();
});

$("chatInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
});

document.querySelectorAll(".quick-questions button").forEach(button => {
    button.addEventListener("click", () => {
        sendChatMessage(button.dataset.question);
    });
});
