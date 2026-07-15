"use strict";

const statusEndpoint = "/api/status";
const refreshIntervalMs = 1000;

const elements = {
    totalSlots: document.getElementById("total-slots"),
    occupiedSlots: document.getElementById("occupied-slots"),
    availableSlots: document.getElementById("available-slots"),
    occupancyPercentage: document.getElementById("occupancy-percentage"),
    processingFps: document.getElementById("processing-fps"),
    connectionStatus: document.getElementById("connection-status"),
    currentTime: document.getElementById("current-time"),
    videoStream: document.getElementById("video-stream"),
    videoFrame: document.querySelector(".video-frame")
};

function formatNumber(value, fallback = "--") {
    const number = Number(value);
    return Number.isFinite(number) ? String(Math.round(number)) : fallback;
}

function formatPercentage(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(1)}%` : "--%";
}

function formatFps(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(1)} FPS` : "-- FPS";
}

function setText(element, value) {
    if (element) {
        element.textContent = value;
    }
}

function setConnectionState(isOnline) {
    if (!elements.connectionStatus) {
        return;
    }

    elements.connectionStatus.classList.toggle("disconnected", !isOnline);
    elements.connectionStatus.innerHTML = isOnline
        ? '<span class="pulse-dot" aria-hidden="true"></span>ONLINE'
        : '<span class="pulse-dot" aria-hidden="true"></span>Disconnected';
}

function updateStatistics(data) {
    setText(elements.totalSlots, formatNumber(data.total_slots));
    setText(elements.occupiedSlots, formatNumber(data.occupied_slots));
    setText(elements.availableSlots, formatNumber(data.available_slots));
    setText(elements.occupancyPercentage, formatPercentage(data.occupancy_percentage));
    setText(elements.processingFps, formatFps(data.fps));
}

async function fetchStatus() {
    try {
        const response = await fetch(statusEndpoint, {
            cache: "no-store",
            headers: {
                Accept: "application/json"
            }
        });

        if (!response.ok) {
            throw new Error(`Status request failed: ${response.status}`);
        }

        const data = await response.json();
        updateStatistics(data);
        setConnectionState(true);
    } catch (error) {
        setConnectionState(false);
        setText(elements.processingFps, "-- FPS");
        console.warn("Status update failed", error);
    }
}

function updateCurrentTime() {
    const now = new Date();
    const formattedTime = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });

    setText(elements.currentTime, formattedTime);
}

function showVideoError() {
    if (elements.videoFrame) {
        elements.videoFrame.classList.add("stream-error");
    }
}

function hideVideoError() {
    if (elements.videoFrame) {
        elements.videoFrame.classList.remove("stream-error");
    }
}

function initializeVideoHandlers() {
    if (!elements.videoStream) {
        return;
    }

    elements.videoStream.addEventListener("load", hideVideoError);
    elements.videoStream.addEventListener("error", showVideoError);
}

function startDashboard() {
    initializeVideoHandlers();
    updateCurrentTime();
    fetchStatus();

    window.setInterval(updateCurrentTime, refreshIntervalMs);
    window.setInterval(fetchStatus, refreshIntervalMs);
}

document.addEventListener("DOMContentLoaded", startDashboard);
