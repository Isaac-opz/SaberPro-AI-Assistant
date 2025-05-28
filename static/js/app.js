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

function displayMessage(text, role) {
    // Añade un mensaje (user, assistant, error, info) al contenedor visible
    const lineDiv = document.createElement('div');
    lineDiv.classList.add('line');
    lineDiv.classList.add(role); // role puede ser 'user', 'server', 'error', 'info'
    // Escapa HTML básico para seguridad simple
    lineDiv.innerHTML = text.replace(/\n/g, "<br/>");
    lines.appendChild(lineDiv);
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
    displayMessage(txt, 'user');

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
        console.debug("Mensaje WebSocket recibido:", rdata); // Para depuración

        let lastServerLine = null;
        const serverLines = lines.querySelectorAll(".line.server");
        if (serverLines.length > 0) {
            lastServerLine = serverLines[serverLines.length - 1];
        }

        switch (rdata.action) {
            case "init_system_response":
                loadingbar.style.display = "block";
                // Crea el div para la respuesta del servidor (vacío inicialmente)
                displayMessage("", "server");
                // Actualiza el div del servidor con el nuevo mensaje
                linesData.push({ "role": "assistant", "content": "" });
                break;

            case "append_system_response":
                if (lastServerLine) {
                    // Añade contenido al último div del servidor
                    // Aquí asumimos que el backend envía texto plano o texto con \n.
                    lastServerLine.innerHTML += rdata.content.replace(/\n/g, "<br/>");
                    // Actualiza el historial interno
                    if (linesData.length > 0 && linesData[linesData.length - 1].role === "assistant") {
                        linesData[linesData.length - 1].content += rdata.content;
                    }
                    scrollToBottom();
                } else {
                    console.warn("Recibido 'append' pero no hay línea de servidor previa.");
                }
                break;

            case "finish_system_response":
                loadingbar.style.display = "none";
                break;

            case "error":
                console.error("Error del Servidor:", rdata.message);
                displayMessage(`Error del servidor: ${rdata.message}`, "error");
                loadingbar.style.display = "none"; // Ocultar barra de carga en error
                break;

            case "info": // Manejo de mensajes informativos del backend
                console.info("Info del Servidor:", rdata.message);
                displayMessage(rdata.message, "info");
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
    // Determina el protocolo correcto (ws:// o wss://)
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}${url}`; // Usa window.location.host para obtener el host actual

    console.log(`Intentando conectar WebSocket a: ${wsUrl}`);

    // Crea la nueva instancia de WebSocket
    const currentSocket = new WebSocket(wsUrl);

    currentSocket.addEventListener("open", (event) => {
        console.log("Conexión WebSocket abierta.");
    });

    currentSocket.addEventListener("close", (event) => {
        console.warn(`Conexión WebSocket cerrada (Código: ${event.code}, Razón: ${event.reason || 'N/A'}). Intentando reconectar en 3 segundos...`);
        // Reconectar después de un retraso de 3 segundos
        setTimeout(() => {
            // Llama de nuevo a openSocket para intentar establecer una nueva conexión
            // La variable global 'socket' se actualiza si la conexión es exitosa
            // Si la conexión falla, se intenta otra vez a los 3 segundos
            console.log("Intentando reconectar...");
            socket = openSocket(url); // Intenta reconectar y actualiza la variable global 
        }, 3000); // Esperar 3 segundos (3000 ms) antes de intentar reconectar
    });

    currentSocket.addEventListener("message", (event) => {
        // Procesa los mensajes recibidos
        processMessage(event);
    });

    currentSocket.addEventListener("error", (event) => {
        console.error("Error de WebSocket observado:", event);
    });

    return currentSocket; // Retorna la instancia creada
}

function displayMessage(text, role) {
    const lineDiv = document.createElement('div');
    lineDiv.classList.add('line', role);
    lineDiv.innerHTML = text.replace(/\n/g, "<br/>");
    lines.appendChild(lineDiv);
    lineDiv.style.opacity = 0;
    setTimeout(() => {
        lineDiv.style.opacity = 1;
    }, 10);
    scrollToBottom();
    return lineDiv;
}

// --- Inicialización ---
socket = openSocket("/init");