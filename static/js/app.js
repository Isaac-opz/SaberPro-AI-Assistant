// --- Elementos del DOM ---
var chatinput = document.getElementById("chatinput");
var lines = document.getElementById("lines");
var linesContainer = document.getElementById("linescontainer"); // Contenedor para scroll
var loadingbar = document.getElementById("loadingbar");

// --- Estado del Chat ---
var linesData = []; // Historial de mensajes para enviar al backend
var socket = null; // Variable para mantener la instancia del WebSocket

// --- Funciones Auxiliares ---
function scrollToBottom() {
    linesContainer.scrollTop = linesContainer.scrollHeight;
}

/**
 * Convierte URLs en texto a etiquetas <a> clickables.
 * Maneja (Link:...), http(s)://, www., y formatos especiales de YouTube.
 * @param {string} text - El texto a procesar.
 * @returns {string} - El texto con URLs convertidas a hipervínculos.
 */
function linkify(text) {
    // Manejar el patrón "Link:..."
    // Ejemplos de Link: (Link:https://www.icfes.gov.co/...), (Link:www.icfes.gov.co), (Link:youtube/videoID?si=PARAM)
    const linkPattern = /\(Link:([^)]+)\)/g;
    text = text.replace(linkPattern, (match, capturedUrl) => {
        let url = capturedUrl.trim();
        let sUrl = url;

        // Anteponer http:// si no tiene esquema y no es un atajo de youtube/
        if (!sUrl.match(/^[a-zA-Z]+:\/\//i)) {
            if (sUrl.toLowerCase().startsWith("youtube/")) {
                // Manejar youtube/videoID o youtube/videoID?params
                const parts = sUrl.split('?');
                const videoPath = parts[0];
                const queryParams = parts.length > 1 ? '?' + parts[1] : '';
                sUrl = `https://www.youtube.com/watch?v=${videoPath.substring(8)}${queryParams.replace(/^\?/, '&')}`;
            } else {
                sUrl = 'http://' + sUrl;
            }
        } else if (sUrl.match(/https?:\/\/(?:www\.)?youtube\/([a-zA-Z0-9_-]+)/i) && !sUrl.includes("watch?v=")) {
            // Corregir https://youtube/VIDEOID a https://www.youtube.com/watch?v=VIDEOID
            sUrl = sUrl.replace(/https?:\/\/(?:www\.)?youtube\/([a-zA-Z0-9_-]+)(\?.*)?/i, (m, videoId, queryParamsStr) => {
                return `https://www.youtube.com/watch?v=${videoId}${queryParamsStr ? queryParamsStr.replace(/^\?/, '&') : ''}`;
            });
        }

        return `<a href="${sUrl}" target="_blank" rel="noopener noreferrer">${url}</a>`;
    });

    // Manejar URLs generales (http, https, www) que no hayan sido ya enlazadas.
    // Intenta evitar reenlazar lo que ya está en una etiqueta <a> o falsos positivos comunes.
    // Busca URLs que no estén precedidas por "href=" o ">".
    const urlPattern = /(?<!href=["'])(?<!>)\b((?:https?|ftp):\/\/[-A-Z0-9+&@#\/%?=~_|$!:,.;]*[A-Z0-9+&@#\/%=~_|$])|(?<!href=["'])(?<!>)\b(www\.[-A-Z0-9+&@#\/%?=~_|$!:,.;]*[A-Z0-9+&@#\/%=~_|$])|(?<!href=["'])(?<![.\/])(?<!@)\b([a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|co|info|biz|io|me|tv|cc|ca|uk|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|nz|cn|br|kr|mx|sa|ae|in|sg|hk|tw|vn|th|ph|id|my|ar|cl|pe|ec|bo|uy|py|ve|cr|pa|do|gt|hn|ni|sv|bz|jm|cu|pr|bb|lc|vc|gd|ag|kn|ai|ms|tc|vg|bm|ky|aw|gp|mq|gf|sr|gy|fk)\b(?:\/(?:[-A-Z0-9+&@#\/%?=~_|$!:,.;]*[A-Z0-9+&@#\/%=~_|$])?)?)(?![^<>]*>|[^<]*<\/a>)/gi;

    text = text.replace(urlPattern, (match, httpUrl, wwwUrl, domainUrl) => {
        let fullMatch = httpUrl || wwwUrl || domainUrl;
        if (!fullMatch) return match;

        // Heurística para evitar reenlazar el texto de un enlace ya creado por "Link:..."
        if (text.includes(`<a href[^>]+>${fullMatch}</a>`)) {
            return fullMatch;
        }

        let urlToLink = fullMatch;
        if (wwwUrl || (domainUrl && !httpUrl)) {
            urlToLink = 'http://' + fullMatch;
        }
        return `<a href="${urlToLink}" target="_blank" rel="noopener noreferrer">${fullMatch}</a>`;
    });

    return text;
}


function displayMessage(text, role) {
    const lineDiv = document.createElement('div');
    lineDiv.classList.add('line', role); // role puede ser 'user', 'server', 'error', 'info'

    let contentToDisplay = text;
    // Linkify solo para mensajes de error o info provenientes del servidor.
    // Los mensajes 'server' (asistente) se linkifican en processMessage, pues se reciben por streaming.
    // Los mensajes 'user' se muestran tal cual.
    if (role === 'error' || role === 'info') {
        contentToDisplay = linkify(text);
    }

    lineDiv.innerHTML = contentToDisplay.replace(/\n/g, "<br/>");
    lines.appendChild(lineDiv);

    // Animación de aparición (fade-in)
    lineDiv.style.opacity = 0;
    setTimeout(() => {
        lineDiv.style.opacity = 1;
    }, 10);

    scrollToBottom();
    return lineDiv;
}

// --- Lógica Principal del Chat ---
function submitText() {
    var txt = chatinput.innerText.trim();
    if (!txt) {
        return false; // No enviar mensajes vacíos
    }
    chatinput.innerText = ""; // Limpiar input

    // Muestra el mensaje del usuario en la UI
    displayMessage(txt, 'user'); // Los mensajes del usuario no se linkifican aquí

    // Añade al historial que se enviará al backend
    linesData.push({ "role": "user", "content": txt });

    // Envía el historial completo al backend
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(linesData));
    } else {
        console.error("WebSocket no está conectado. No se puede enviar el mensaje.");
        displayMessage("Error: No se pudo conectar con el servidor.", "error");
    }
    return false; // Previene el envío del formulario por defecto
}

function processMessage(event) {
    try {
        const rdata = JSON.parse(event.data);
        console.debug("Mensaje WebSocket recibido:", rdata); // depuración

        const serverLines = lines.querySelectorAll(".line.server");
        let lastServerLine = serverLines.length > 0 ? serverLines[serverLines.length - 1] : null;

        switch (rdata.action) {
            case "init_system_response":
                loadingbar.style.display = "block";
                // Crea el div para la respuesta del servidor (vacío inicialmente)
                // displayMessage crea el div, pero el contenido se llena en append
                if (!lastServerLine || linesData.length === 0 || linesData[linesData.length - 1].role !== "assistant" || linesData[linesData.length - 1].content !== "") {
                    displayMessage("", "server"); // Crea un nuevo div para el mensaje del servidor
                    linesData.push({ "role": "assistant", "content": "" });
                }
                lastServerLine = lines.querySelectorAll(".line.server")[lines.querySelectorAll(".line.server").length - 1]; // re-fetch lastServerLine
                break;

            case "append_system_response":
                if (linesData.length > 0 && linesData[linesData.length - 1].role === "assistant") {
                    // Acumula el contenido raw en linesData
                    linesData[linesData.length - 1].content += rdata.content;

                    if (lastServerLine) {
                        // Obtiene el mensaje completo del asistente
                        const fullAssistantMessage = linesData[linesData.length - 1].content;
                        // Linkifica el mensaje completo y luego reemplaza saltos de línea
                        lastServerLine.innerHTML = linkify(fullAssistantMessage).replace(/\n/g, "<br/>");
                        scrollToBottom();
                    } else {
                        console.warn("Recibido 'append' pero no hay línea de servidor previa (lastServerLine es null).");
                        // se asume que 'init' crea la línea de servidor
                    }
                } else {
                    console.warn("Recibido 'append' pero no hay mensaje de asistente en linesData.");
                }
                break;

            case "finish_system_response":
                loadingbar.style.display = "none";
                if (linesData.length > 0 && linesData[linesData.length - 1].role === "assistant" && lastServerLine) {
                    const fullAssistantMessage = linesData[linesData.length - 1].content;
                    const finalHtml = linkify(fullAssistantMessage).replace(/\n/g, "<br/>");
                    if (lastServerLine.innerHTML !== finalHtml) {
                        lastServerLine.innerHTML = finalHtml;
                        scrollToBottom();
                    }
                }
                break;

            case "error":
                console.error("Error del Servidor:", rdata.message);
                displayMessage(`Error del servidor: ${rdata.message}`, "error"); // Se linkificará si el mensaje de error contiene URLs
                loadingbar.style.display = "none";
                break;

            case "info":
                console.info("Info del Servidor:", rdata.message);
                displayMessage(rdata.message, "info"); // Se linkificará si el mensaje de info contiene URLs
                loadingbar.style.display = "none";
                break;

            default:
                console.warn("Acción desconocida recibida:", rdata.action);
        }
    } catch (e) {
        console.error("Error al procesar mensaje WebSocket:", e);
        console.error("Dato recibido:", event.data);
        displayMessage("Error procesando la respuesta del servidor.", "error");
        loadingbar.style.display = "none";
    }
}

// --- Conexión WebSocket ---
function openSocket(url) {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}${url}`;

    console.log(`Intentando conectar WebSocket a: ${wsUrl}`);
    const currentSocket = new WebSocket(wsUrl);

    currentSocket.addEventListener("open", (event) => {
        console.log("Conexión WebSocket abierta.");
    });

    currentSocket.addEventListener("close", (event) => {
        console.warn(`Conexión WebSocket cerrada (Código: ${event.code}, Razón: ${event.reason || 'N/A'}). Intentando reconectar en 3 segundos...`);
        setTimeout(() => {
            console.log("Intentando reconectar...");
            socket = openSocket(url);
        }, 3000);
    });

    currentSocket.addEventListener("message", (event) => {
        processMessage(event);
    });

    currentSocket.addEventListener("error", (event) => {
        console.error("Error de WebSocket observado:", event);
    });

    return currentSocket;
}

// --- Inicialización ---
socket = openSocket("/init");