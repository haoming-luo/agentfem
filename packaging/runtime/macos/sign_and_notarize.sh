#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 PACKAGE.pkg NOTARY_KEYCHAIN_PROFILE" >&2
  exit 2
fi

package=$1
profile=$2

pkgutil --check-signature "$package"
xcrun notarytool submit "$package" --keychain-profile "$profile" --wait
xcrun stapler staple "$package"
xcrun stapler validate "$package"
spctl --assess --type install --verbose=4 "$package"

