#!/bin/bash

if ! curl --version &> /dev/null
then
    echo "The program curl could not be found. Please install it: https://curl.se/"
    exit 1
fi

set -e

VERSION=$(curl --header 'Accept: text/plain' --silent https://beaker.org/api/v3/release)
TMP=$(mktemp -d)

echo Downloading Beaker CLI version $VERSION.
echo

ARCH=$(uname -p)
OS=$(uname -s)

if [[ $ARCH == "arm" && $OS == "Darwin" ]]; then
  # macOS on M1 chipsets
  ARCH=arm64; OS=darwin
elif [[ $ARCH == "i386" && $OS == "Darwin" ]]; then
  # macOS on Intel chipsets
  ARCH=amd64; OS=darwin
elif [[ $ARCH == "x86_64" && $OS == "Linux" ]]; then
  # Linux on AMD64 chipsets
  ARCH=amd64; OS=linux
else
  echo Unrecognized OS-Architecture combination:
  echo ARCH=$ARCH
  echo OS=$OS
  exit 1
fi

PATTERN="beaker-cli-$OS-$ARCH-$VERSION.tar.gz"
mkdir $TMP/assets/

URL="https://beaker.org/api/v3/release/cli?os=$OS&arch=$ARCH"
HTTP_CODE=$(curl --retry 25 --retry-delay 1 --max-time 10 --retry-max-time 10 --silent --output $TMP/assets/$PATTERN --write-out "%{http_code}" $URL)
if [[ "$HTTP_CODE" -ne "200" ]]; then
  echo "Unexpected HTTP code ($HTTP_CODE) when retrieving: $URL"
  echo
  echo Please, try again later.
  exit 1
fi

echo "Download succeeded; size is $(wc -c < $TMP/assets/$PATTERN) bytes."
echo

BIN=$HOME/.local/bin
if test -e $BIN/beaker; then
  echo "The file $BIN/beaker already exists and will be overwritten, is that ok?"
  echo
  echo "Press enter to proceed and Ctrl^C to cancel"
  read
fi

echo "We will try to extract the CLI binary to $BIN/beaker. To do so, we will"
echo "run tar with sudo. You will be prompted for your local password (e.g., your macOS"
echo "password)."
echo
CMD="tar -zxf $TMP/assets/$PATTERN -C $BIN ./beaker"
echo "Command we're running: $CMD"
echo
$CMD

echo "Okay, beaker CLI version $VERSION should now be installed to $BIN/beaker."
echo
echo Testing by running: beaker --version
echo
beaker --version
echo
echo
