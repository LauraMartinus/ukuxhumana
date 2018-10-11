#!/bin/bash
WORKSPACE=/workspace

TEMP_UID="${TEMP_UID:-1000}"
set -ux
useradd -s /bin/falsae --no-create-home -u ${TEMP_UID} temp
exec  "$@"
