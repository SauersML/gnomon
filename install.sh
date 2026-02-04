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
BINARY_NAME="gnomon"
# Default to user-local install (no sudo required, works everywhere)
INSTALL_DIR="$HOME/.local/bin"

log_info() { echo -e "${ICON_INFO}  $1"; }
log_success() { echo -e "${ICON_CHECK}  $1"; }
log_error() { echo -e "${ICON_CROSS}  ${RED}$1${RESET}"; }
log_header() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }

# --- 1. Detect Platform ---
log_header "Detecting Platform"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"

if [[ "$OS" == mingw* ]] || [[ "$OS" == msys* ]] || [[ "$OS" == cygwin* ]]; then
    log_info "Raw Git Bash detection (uname): $OS / $ARCH"
    log_info "Windows Env: PROCESSOR_ARCHITECTURE='$PROCESSOR_ARCHITECTURE', PROCESSOR_ARCHITEW6432='$PROCESSOR_ARCHITEW6432'"
fi

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
        RAW_OS="$OS"
        OS="windows"
        INSTALL_DIR="$HOME/bin"
        
        # Check for Windows ARM64 environment variables (Git Bash often runs as x64 emulated)
        # Fallback: check if uname string contains "arm64" (e.g. mingw64_nt-10.0-26200-arm64)
        if [[ "$PROCESSOR_ARCHITECTURE" == "ARM64" ]] || \
           [[ "$PROCESSOR_ARCHITEW6432" == "ARM64" ]] || \
           [[ "$RAW_OS" == *"arm64"* ]]; then
            ARCH="aarch64"
            log_info "Detected Windows ARM64 environment."
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

# --- 2. Find Latest Binary Release ---
log_header "Checking Latest Binary Release"

# Fetch all releases to find the most recent one with binary assets
# This skips model-only releases (e.g., models-v1) automatically
API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases"

CURL_ARGS=(-sL --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 5 --max-time 20)
if [ -n "$GITHUB_TOKEN" ]; then
    log_info "Using GITHUB_TOKEN for authenticated API request."
    CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
elif [ -n "$GH_TOKEN" ]; then
    log_info "Using GH_TOKEN for authenticated API request."
    CURL_ARGS+=(-H "Authorization: token $GH_TOKEN")
fi

MAX_RETRIES=30
DELAY=5
DOWNLOAD_URL=""

for ((i=1; i<=MAX_RETRIES; i++)); do
    # Only print "Fetching..." on first attempt or every 5th retry to reduce log noise
    if [ $i -eq 1 ] || [ $((i % 5)) -eq 0 ]; then
        log_info "Fetching releases from GitHub (Attempt $i/$MAX_RETRIES)..."
    fi

    RESPONSE=$(curl "${CURL_ARGS[@]}" "${API_URL}" || true)

    # Find the first release that contains the target asset
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
    log_error "Could not find a download URL for ${TARGET_ASSET} in any release."
    log_error "GitHub API Response Preview:"
    echo "$RESPONSE" | head -n 20
    log_error "..."
    log_error "Please check https://github.com/${REPO_OWNER}/${REPO_NAME}/releases manually."
    exit 1
fi

log_success "Found latest binary release."

# --- 3. Download & Install ---
log_header "Installing gnomon"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

ARCHIVE_PATH="${TEMP_DIR}/${TARGET_ASSET}"

log_info "${ICON_PKG} Downloading..."
curl -L --progress-bar --retry 5 --retry-delay 3 --retry-connrefused --connect-timeout 10 --max-time 300 -o "$ARCHIVE_PATH" "$DOWNLOAD_URL"

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
         log_error "Failed to create ${INSTALL_DIR}."
         exit 1
    }
fi

mv "$SOURCE_BIN" "${INSTALL_DIR}/${DEST_BINARY_NAME}"
chmod +x "${INSTALL_DIR}/${DEST_BINARY_NAME}"

# --- 4. Verify ---
log_header "Verification"

INSTALLED_BIN="${INSTALL_DIR}/${DEST_BINARY_NAME}"

# Test the binary directly from install path
if [ -x "$INSTALLED_BIN" ] && "$INSTALLED_BIN" --help >/dev/null 2>&1; then
    log_success "Successfully installed gnomon!"
    
    # Check if install dir is in PATH
    if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
        echo -e "\n${ICON_ROCK}  ${BOLD}Run 'gnomon --help' to get started!${RESET}\n"
    else
        # Auto-add to PATH in shell config
        PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
        
        # Determine which shell config to update
        if [ -n "$ZSH_VERSION" ] || [[ "$SHELL" == *"zsh"* ]]; then
            SHELL_RC="$HOME/.zshrc"
        else
            SHELL_RC="$HOME/.bashrc"
        fi
        
        # Check if already in config (avoid duplicates)
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Added by gnomon installer" >> "$SHELL_RC"
            echo "$PATH_LINE" >> "$SHELL_RC"
            log_success "Added ${INSTALL_DIR} to PATH in ${SHELL_RC}"
        fi
        
        echo -e "\n${ICON_ROCK}  ${BOLD}Restart your terminal or run: source ${SHELL_RC}${RESET}"
        echo -e "    Then run: ${BOLD}gnomon --help${RESET}\n"
    fi
else
    log_error "Binary installed but failed to run."
    log_info "Try running: ${INSTALLED_BIN} --help"
    exit 1
fi
