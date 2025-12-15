#!/bin/bash
set -e

# ==============================================================================
#  gnomon Universal Installer
# ==============================================================================
#  Installs the latest release of gnomon for your platform.
#  Repo: https://github.com/SauersML/gnomon
# ==============================================================================

# --- Colors & Styles ---
BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

# --- Icons ---
ICON_CHECK="${GREEN}✔${RESET}"
ICON_CROSS="${RED}✘${RESET}"
ICON_INFO="${CYAN}ℹ${RESET}"
ICON_PKG="${YELLOW}📦${RESET}"
ICON_ROCK="${GREEN}🚀${RESET}"

# --- Configuration ---
REPO_OWNER="SauersML"
REPO_NAME="gnomon"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="gnomon"

log_info() { echo -e "${ICON_INFO}  $1"; }
log_success() { echo -e "${ICON_CHECK}  $1"; }
log_error() { echo -e "${ICON_CROSS}  ${RED}$1${RESET}"; }
log_header() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }

# --- 1. Detect Platform ---
log_header "Detecting Platform"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"

TARGET_ASSET=""

case "$OS" in
    linux)
        case "$ARCH" in
            x86_64)  TARGET_ASSET="gnomon-linux-x64.tar.gz" ;;
            aarch64) TARGET_ASSET="gnomon-linux-arm64.tar.gz" ;;
            *)
                log_error "Unsupported Linux architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    darwin)
        case "$ARCH" in
            x86_64) TARGET_ASSET="gnomon-macos-intel.tar.gz" ;;
            arm64)  TARGET_ASSET="gnomon-macos-arm64.tar.gz" ;;
            *)
                log_error "Unsupported macOS architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    mingw*|msys*|cygwin*)
        OS="windows"
        INSTALL_DIR="$HOME/bin"
        
        # Check for Windows ARM64 environment variables (Git Bash often runs as x64 emulated)
        if [[ "$PROCESSOR_ARCHITECTURE" == "ARM64" ]] || [[ "$PROCESSOR_ARCHITEW6432" == "ARM64" ]]; then
            ARCH="aarch64"
        fi

        case "$ARCH" in
            x86_64) TARGET_ASSET="gnomon-windows-x64.zip" ;;
            aarch64|arm64) TARGET_ASSET="gnomon-windows-arm64.zip" ;;
            *)
                log_error "Unsupported Windows architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    *)
        log_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

log_success "Detected: ${BOLD}${OS}/${ARCH}${RESET}"
log_info "Target release asset: ${BOLD}${TARGET_ASSET}${RESET}"

# --- 2. Find Latest Release ---
log_header "Checking Latest Release"

API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest"
log_info "Fetching metadata from GitHub..."

# Fetch release info, filtering for the asset URL that matches our target name
# We use grep/sed hacks to avoid 'jq' dependency for universal compatibility.
# We retry a few times because GitHub API can be slow to update 'latest' immediately after release.

MAX_RETRIES=30
DELAY=5
DOWNLOAD_URL=""

for ((i=1; i<=MAX_RETRIES; i++)); do
    # Only print "Fetching..." on first attempt or every 5th retry to reduce log noise
    if [ $i -eq 1 ] || [ $((i % 5)) -eq 0 ]; then
        log_info "Fetching metadata from GitHub (Attempt $i/$MAX_RETRIES)..."
    fi
    
    RESPONSE=$(curl -sL "${API_URL}")
    
    DOWNLOAD_URL=$(echo "$RESPONSE" | \
        grep "browser_download_url.*${TARGET_ASSET}" | \
        cut -d '"' -f 4 | \
        head -n 1)

    if [ -n "$DOWNLOAD_URL" ]; then
        break
    fi

    if [ $i -lt $MAX_RETRIES ]; then
        sleep $DELAY
    fi
done

if [ -z "$DOWNLOAD_URL" ]; then
    log_error "Could not find a download URL for ${TARGET_ASSET} in the latest release."
    log_error "Available assets in release:"
    echo "$RESPONSE" | grep "browser_download_url" | cut -d '"' -f 4
    log_error "Please check https://github.com/${REPO_OWNER}/${REPO_NAME}/releases manually."
    exit 1
fi

log_success "Found latest release asset."

# --- 3. Download & Install ---
log_header "Installing gnomon"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

ARCHIVE_PATH="${TEMP_DIR}/${TARGET_ASSET}"

log_info "${ICON_PKG} Downloading..."
curl -L --progress-bar -o "$ARCHIVE_PATH" "$DOWNLOAD_URL"

log_info "Extracting..."
if [[ "$TARGET_ASSET" == *.zip ]]; then
    unzip -q "$ARCHIVE_PATH" -d "$TEMP_DIR"
else
    tar -xzf "$ARCHIVE_PATH" -C "$TEMP_DIR"
fi

# Determine binary location after extraction.
# The binary inside the tarball is named identically to the asset (minus extensions)
# e.g. gnomon-macos-arm64.tar.gz -> gnomon-macos-arm64
if [[ "$TARGET_ASSET" == *.zip ]]; then
    INTERNAL_BIN_NAME="${TARGET_ASSET%.zip}.exe"
    DEST_BINARY_NAME="${BINARY_NAME}.exe"
else
    INTERNAL_BIN_NAME="${TARGET_ASSET%.tar.gz}"
    DEST_BINARY_NAME="${BINARY_NAME}"
fi

# Search for this specific file
SOURCE_BIN=$(find "$TEMP_DIR" -type f -name "${INTERNAL_BIN_NAME}" | head -n 1)

# Fallback: if not found, look for any executable file that is not the archive itself
# For Windows, look for .exe
if [ -z "$SOURCE_BIN" ] || [ ! -f "$SOURCE_BIN" ]; then
    log_info "Exact match not found. Searching for executable..."
    if [[ "$OS" == "windows" ]]; then
         SOURCE_BIN=$(find "$TEMP_DIR" -type f -name "*.exe" | head -n 1)
    else
         SOURCE_BIN=$(find "$TEMP_DIR" -type f -perm +111 ! -name "*.*" | head -n 1)
    fi
fi

if [ -z "$SOURCE_BIN" ] || [ ! -f "$SOURCE_BIN" ]; then
    log_error "Binary not found in extracted archive."
    log_error "Contents of extraction: $(ls -R "$TEMP_DIR")"
    exit 1
fi

log_info "Installing to ${INSTALL_DIR}..."

# Create install dir if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    log_info "Creating directory ${INSTALL_DIR}..."
    mkdir -p "$INSTALL_DIR" || {
         log_error "Failed to create ${INSTALL_DIR}. Try running with sudo."
         exit 1
    }
fi

if [ ! -w "$INSTALL_DIR" ]; then
    if command -v sudo >/dev/null 2>&1; then
        log_info "Sudo permissions required to write to ${INSTALL_DIR}"
        sudo mv "$SOURCE_BIN" "${INSTALL_DIR}/${DEST_BINARY_NAME}"
        sudo chmod +x "${INSTALL_DIR}/${DEST_BINARY_NAME}"
    else
        log_error "Directory ${INSTALL_DIR} is not writable and 'sudo' is not available."
        log_error "Please run this script as Administrator or choose a writable directory."
        exit 1
    fi
else
    mv "$SOURCE_BIN" "${INSTALL_DIR}/${DEST_BINARY_NAME}"
    chmod +x "${INSTALL_DIR}/${DEST_BINARY_NAME}"
fi

# --- 4. Verify ---
log_header "Verification"

if command -v "${BINARY_NAME}" >/dev/null 2>&1; then
    # Binary doesn't support --version, so we check help output
    if "${BINARY_NAME}" --help >/dev/null 2>&1; then
        log_success "Successfully installed gnomon!"
        echo -e "\n${ICON_ROCK}  ${BOLD}Run 'gnomon --help' to get started!${RESET}\n"
    elif [ "$OS" = "windows" ]; then
         # On Windows, newly added path might not be available in current shell session immediately
         log_success "Successfully installed gnomon to ${INSTALL_DIR}."
         echo -e "Note: You may need to restart your terminal or run: export PATH=\"\$HOME/bin:\$PATH\""
    else
        log_error "Binary installed but failed to run."
        exit 1
    fi
else
    log_error "Installation appeared to succeed, but 'gnomon' is not in your PATH."
    log_info "Ensure ${INSTALL_DIR} is in your PATH."
    exit 1
fi
