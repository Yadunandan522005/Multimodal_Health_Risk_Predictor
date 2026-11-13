/* =========================================================
   HEALTHRISK AI
   MAIN PROJECT JAVASCRIPT
========================================================= */

"use strict";


/* =========================================================
   WAIT UNTIL PAGE IS READY
========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    initializeApplication();

});



/* =========================================================
   MAIN INITIALIZATION
========================================================= */

function initializeApplication() {

    setupAnalyzeAll();

    setupFactorCards();

    setupModalButtons();

    setupNavigationSafety();

}



/* =========================================================
   ANALYZE ALL
========================================================= */

function setupAnalyzeAll() {

    /*
       Supports all of these possible buttons:

       id="analyzeAll"

       id="analyze-all"

       class="analyze-all"

       class="analyze-all-btn"

       data-action="analyze-all"
    */

    const buttons = document.querySelectorAll(
        "#analyzeAll, " +
        "#analyze-all, " +
        ".analyze-all, " +
        ".analyze-all-btn, " +
        "[data-action='analyze-all']"
    );


    buttons.forEach(function (button) {

        button.addEventListener("click", function (event) {

            event.preventDefault();

            goToPredictionPage();

        });

    });

}



/* =========================================================
   GLOBAL ANALYZE ALL FUNCTION
   This also supports onclick="analyzeAll()"
========================================================= */

window.analyzeAll = function () {

    goToPredictionPage();

};



function goToPredictionPage() {

    window.location.href = "/predict";

}



/* =========================================================
   FACTOR DEFINITIONS
========================================================= */

const factorDefinitions = {

    laboratory: {

        title: "Laboratory Reports",

        description:
            "Enter laboratory and blood test information.",

        icon:
            "fa-solid fa-flask-vial"

    },


    history: {

        title: "Medical History",

        description:
            "Enter patient medical and family history.",

        icon:
            "fa-solid fa-file-medical"

    },


    vital: {

        title: "Vital Signs",

        description:
            "Enter the patient's vital health measurements.",

        icon:
            "fa-solid fa-heart-pulse"

    },


    imaging: {

        title: "Medical Imaging",

        description:
            "Enter medical imaging and diagnostic findings.",

        icon:
            "fa-solid fa-x-ray"

    }

};



/* =========================================================
   FACTOR CARDS
========================================================= */

function setupFactorCards() {

    const cards = document.querySelectorAll(

        "[data-factor], " +
        ".factor-card, " +
        ".input-factor-card, " +
        ".health-factor-card"

    );


    cards.forEach(function (card) {

        card.addEventListener("click", function (event) {

            /*
               Do not interfere with normal links.
            */

            if (
                event.target.closest("a") &&
                event.target.closest("a").getAttribute("href")
            ) {

                return;

            }


            let factor =
                card.dataset.factor ||
                card.dataset.type ||
                card.getAttribute("data-modal");


            if (!factor) {

                const text =
                    card.innerText.toLowerCase();


                if (
                    text.includes("laboratory")
                ) {

                    factor = "laboratory";

                }

                else if (
                    text.includes("medical history")
                ) {

                    factor = "history";

                }

                else if (
                    text.includes("vital")
                ) {

                    factor = "vital";

                }

                else if (
                    text.includes("medical imaging") ||
                    text.includes("imaging")
                ) {

                    factor = "imaging";

                }

            }


            if (factor) {

                openFactorModal(factor);

            }

        });

    });

}



/* =========================================================
   GLOBAL FACTOR MODAL FUNCTION
   Supports:

   onclick="openFactorModal('laboratory')"
   onclick="openFactorModal('history')"
   onclick="openFactorModal('vital')"
   onclick="openFactorModal('imaging')"
========================================================= */

window.openFactorModal = function (factor) {

    /*
       Normalize names.
    */

    factor =
        String(factor || "")
        .toLowerCase()
        .trim()
        .replace(/[\s_-]+/g, "");


    if (
        factor === "lab" ||
        factor === "laboratoryreports" ||
        factor === "laboratory"
    ) {

        factor = "laboratory";

    }


    else if (
        factor === "medicalhistory" ||
        factor === "history"
    ) {

        factor = "history";

    }


    else if (
        factor === "vitalsigns" ||
        factor === "vital"
    ) {

        factor = "vital";

    }


    else if (
        factor === "medicalimaging" ||
        factor === "imaging"
    ) {

        factor = "imaging";

    }


    const definition =
        factorDefinitions[factor];


    if (!definition) {

        console.warn(
            "Unknown health factor:",
            factor
        );

        return;

    }


    /*
       First look for an existing modal.
    */

    let modal =
        document.getElementById("factorModal");


    /*
       If the modal already exists in index.html,
       use it.
    */

    if (modal) {

        prepareExistingModal(
            modal,
            factor,
            definition
        );

        return;

    }


    /*
       If no modal exists, create one automatically.
       This makes the system robust even if the HTML
       modal was accidentally removed.
    */

    modal =
        createFactorModal(
            factor,
            definition
        );


    document.body.appendChild(modal);


    showModal(modal);

};



/* =========================================================
   PREPARE EXISTING MODAL
========================================================= */

function prepareExistingModal(
    modal,
    factor,
    definition
) {

    /*
       Store currently selected factor.
    */

    modal.dataset.factor = factor;


    /*
       Update title if matching elements exist.
    */

    const title =
        modal.querySelector(
            "#factorModalTitle, " +
            ".factor-modal-title, " +
            ".modal-title"
        );


    if (title) {

        title.textContent =
            definition.title;

    }


    const description =
        modal.querySelector(
            "#factorModalDescription, " +
            ".factor-modal-description, " +
            ".modal-description"
        );


    if (description) {

        description.textContent =
            definition.description;

    }


    const icon =
        modal.querySelector(
            "#factorModalIcon, " +
            ".factor-modal-icon"
        );


    if (icon) {

        icon.className =
            definition.icon +
            " factor-modal-icon";

    }


    showModal(modal);

}



/* =========================================================
   CREATE MODAL IF MISSING
========================================================= */

function createFactorModal(
    factor,
    definition
) {

    const overlay =
        document.createElement("div");


    overlay.id =
        "factorModal";


    overlay.className =
        "factor-modal-overlay";


    overlay.dataset.factor =
        factor;


    overlay.innerHTML = `

        <div class="factor-modal-box">

            <button
                type="button"
                class="factor-modal-close"
                onclick="closeFactorModal()"
                aria-label="Close"
            >
                ×
            </button>


            <div class="factor-modal-icon-wrapper">

                <i
                    class="${definition.icon} factor-modal-icon"
                ></i>

            </div>


            <h2 id="factorModalTitle">
                ${definition.title}
            </h2>


            <p id="factorModalDescription">
                ${definition.description}
            </p>


            <div class="factor-modal-message">

                <i class="fa-solid fa-circle-info"></i>

                <span>
                    To perform the complete health analysis,
                    use <strong>Analyze All</strong> and enter
                    the required information on the prediction page.
                </span>

            </div>


            <div class="factor-modal-actions">

                <button
                    type="button"
                    class="modal-cancel-button"
                    onclick="closeFactorModal()"
                >
                    Close
                </button>


                <button
                    type="button"
                    class="modal-predict-button"
                    onclick="goToPredictionPage()"
                >
                    Continue to Prediction
                </button>

            </div>

        </div>

    `;


    /*
       Clicking outside the box closes the modal.
    */

    overlay.addEventListener(
        "click",
        function (event) {

            if (
                event.target === overlay
            ) {

                closeFactorModal();

            }

        }
    );


    return overlay;

}



/* =========================================================
   SHOW MODAL
========================================================= */

function showModal(modal) {

    modal.style.display =
        "flex";


    document.body.classList.add(
        "modal-open"
    );


    /*
       Small delay allows CSS transition.
    */

    requestAnimationFrame(
        function () {

            modal.classList.add(
                "show"
            );

        }
    );

}



/* =========================================================
   CLOSE MODAL
========================================================= */

window.closeFactorModal = function () {

    const modal =
        document.getElementById(
            "factorModal"
        );


    if (!modal) {

        return;

    }


    modal.classList.remove(
        "show"
    );


    document.body.classList.remove(
        "modal-open"
    );


    setTimeout(
        function () {

            modal.style.display =
                "none";

        },
        180
    );

};



/* =========================================================
   MODAL BUTTON SETUP
========================================================= */

function setupModalButtons() {

    /*
       Close buttons
    */

    document.addEventListener(
        "click",
        function (event) {

            const closeButton =
                event.target.closest(
                    "[data-close-modal], " +
                    ".close-factor-modal, " +
                    ".modal-close"
                );


            if (closeButton) {

                event.preventDefault();

                closeFactorModal();

            }

        }
    );

}



/* =========================================================
   ESCAPE KEY
========================================================= */

document.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Escape"
        ) {

            closeFactorModal();

        }

    }
);



/* =========================================================
   NAVIGATION SAFETY
========================================================= */

function setupNavigationSafety() {

    /*
       Make sure normal navigation links are not
       accidentally blocked by JavaScript.
    */

    const links =
        document.querySelectorAll(
            ".nav-box, .brand"
        );


    links.forEach(function (link) {

        link.addEventListener(
            "click",
            function () {

                /*
                   Intentionally empty.

                   We do NOT call preventDefault().
                   This allows Flask routes to work.
                */

            }
        );

    });

}



/* =========================================================
   VALIDATE MODAL
========================================================= */

window.validateModal = function () {

    /*
       If a form exists inside the modal,
       validate required fields.

       Otherwise simply continue.
    */

    const modal =
        document.getElementById(
            "factorModal"
        );


    if (!modal) {

        return true;

    }


    const requiredFields =
        modal.querySelectorAll(
            "[required]"
        );


    let valid = true;


    requiredFields.forEach(
        function (field) {

            if (
                !field.value ||
                !field.value.trim()
            ) {

                field.classList.add(
                    "input-error"
                );

                valid = false;

            }

            else {

                field.classList.remove(
                    "input-error"
                );

            }

        }
    );


    return valid;

};



/* =========================================================
   OPTIONAL DASHBOARD CHART
========================================================= */

window.initializeDashboardChart =
    function () {

        /*
           This function is intentionally safe.

           If Chart.js or the chart element does not
           exist, nothing happens.
        */

        if (
            typeof Chart === "undefined"
        ) {

            return;

        }


        const canvas =
            document.getElementById(
                "diseasePieChart"
            );


        if (!canvas) {

            return;

        }


        if (
            typeof window.probabilityData ===
            "undefined"
        ) {

            return;

        }


        const probabilityData =
            window.probabilityData;


        const labels =
            Object.keys(
                probabilityData
            );


        const values =
            Object.values(
                probabilityData
            ).map(
                value =>
                    Number(value) * 100
            );


        const colors =
            labels.map(
                function (label) {

                    if (
                        label === "Diabetes"
                    ) {

                        return "#4F8EF7";

                    }


                    if (
                        label === "Heart Disease"
                    ) {

                        return "#F05D7B";

                    }


                    if (
                        label === "Bronchitis"
                    ) {

                        return "#F39A3D";

                    }


                    if (
                        label === "Breast Cancer"
                    ) {

                        return "#F5C84B";

                    }


                    return "#9AA6B2";

                }
            );


        new Chart(
            canvas,
            {

                type: "pie",

                data: {

                    labels: labels,

                    datasets: [

                        {

                            data: values,

                            backgroundColor:
                                colors,

                            borderColor:
                                "#ffffff",

                            borderWidth: 3

                        }

                    ]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio:
                        true,

                    plugins: {

                        legend: {

                            display: false

                        }

                    }

                }

            }
        );

    };



/* =========================================================
   MODAL CSS
   Injected automatically so factor modals work even
   if the main CSS does not contain modal styling.
========================================================= */

(function addModalStyles() {

    if (
        document.getElementById(
            "healthrisk-modal-styles"
        )
    ) {

        return;

    }


    const style =
        document.createElement("style");


    style.id =
        "healthrisk-modal-styles";


    style.textContent = `

        body.modal-open {
            overflow: hidden;
        }


        .factor-modal-overlay {

            position: fixed;

            inset: 0;

            z-index: 99999;

            display: none;

            align-items: center;

            justify-content: center;

            padding: 20px;

            background:
                rgba(18, 39, 62, 0.45);

            backdrop-filter:
                blur(3px);

            opacity: 0;

            transition:
                opacity 0.18s ease;

        }


        .factor-modal-overlay.show {

            opacity: 1;

        }


        .factor-modal-box {

            width: 100%;

            max-width: 470px;

            position: relative;

            padding: 28px;

            border-radius: 18px;

            background: #ffffff;

            border:
                1px solid #dce5f0;

            box-shadow:
                0 20px 60px
                rgba(21, 48, 76, 0.22);

            transform:
                translateY(10px)
                scale(0.98);

            transition:
                transform 0.18s ease;

            text-align: left;

        }


        .factor-modal-overlay.show
        .factor-modal-box {

            transform:
                translateY(0)
                scale(1);

        }


        .factor-modal-close {

            position: absolute;

            top: 12px;

            right: 14px;

            width: 30px;

            height: 30px;

            border: none;

            border-radius: 50%;

            background: #f1f5f9;

            color: #64748b;

            font-size: 20px;

            cursor: pointer;

            line-height: 1;

        }


        .factor-modal-close:hover {

            background: #e6edf5;

            color: #1e3a5f;

        }


        .factor-modal-icon-wrapper {

            width: 48px;

            height: 48px;

            display: flex;

            align-items: center;

            justify-content: center;

            border-radius: 13px;

            background: #edf5ff;

            color: #2879df;

            margin-bottom: 14px;

        }


        .factor-modal-icon {

            font-size: 20px;

        }


        .factor-modal-box h2 {

            margin: 0 0 6px;

            color: #173451;

            font-size: 20px;

        }


        .factor-modal-box > p {

            margin: 0 0 18px;

            color: #718096;

            font-size: 12px;

            line-height: 1.6;

        }


        .factor-modal-message {

            display: flex;

            gap: 9px;

            align-items: flex-start;

            padding: 13px;

            border-radius: 10px;

            background: #f5f9ff;

            border: 1px solid #dce9f8;

            color: #5c7189;

            font-size: 11px;

            line-height: 1.6;

        }


        .factor-modal-message i {

            color: #2879df;

            margin-top: 2px;

        }


        .factor-modal-actions {

            display: flex;

            justify-content: flex-end;

            gap: 8px;

            margin-top: 20px;

        }


        .modal-cancel-button,
        .modal-predict-button {

            border-radius: 8px;

            padding: 9px 14px;

            font-size: 11px;

            font-weight: 600;

            cursor: pointer;

        }


        .modal-cancel-button {

            background: #ffffff;

            border:
                1px solid #d5e0eb;

            color: #47647f;

        }


        .modal-predict-button {

            background: #347ff0;

            border:
                1px solid #347ff0;

            color: #ffffff;

        }


        .input-error {

            border-color:
                #e05262 !important;

            box-shadow:
                0 0 0 2px
                rgba(224,82,98,0.10);

        }


        @media(max-width:600px) {

            .factor-modal-box {

                padding: 22px;

            }

            .factor-modal-actions {

                flex-direction: column;

            }

            .modal-cancel-button,
            .modal-predict-button {

                width: 100%;

            }

        }

    `;


    document.head.appendChild(style);

})();