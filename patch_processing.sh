#!/bin/bash

### TBar patch extraction script

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"; exit 0
fi

TBAR_DIR=$(pwd)
BUG_NAME=$1
D4J_REPO_ROOT="$TBAR_DIR/D4J/projects/"
D4J_TOOL_ROOT="$TBAR_DIR/D4J/defects4j/"
PATCH_SRC_ROOT="$TBAR_DIR/OUTPUT/FixPatterns/TBar"
PATCH_TAR_ROOT="$TBAR_DIR/OUTPUT/PerfectFL/TBar/AllPatches/$BUG_NAME"

## Step 1. Generate Patches.
./PerfectFLTBarRunner.sh $D4J_REPO_ROOT $BUG_NAME $D4J_TOOL_ROOT true

## Step 2. Copy patches to single directory.
mkdir -p $PATCH_TAR_ROOT
find $PATCH_SRC_ROOT -type f -iregex ".*$BUG_NAME\/Patch.*\.txt" -exec cp {} $PATCH_TAR_ROOT \;

## Step 3. Find and copy appropriate java file.
MOD_JAVA_PATH=`find $PATCH_TAR_ROOT -iregex ".*_1_.*\.txt" -exec cat {} \; | head -3 | tail -1 | cut -c7-`
cp "$D4J_REPO_ROOT/$BUG_NAME/$MOD_JAVA_PATH" $PATCH_TAR_ROOT

## Step 4. SrcML-converting script
# remove previous stuff and find appropriate java file
cd $PATCH_TAR_ROOT
rm *txt.java*
cd $TBAR_DIR
JAVAFILE=`find $PATCH_TAR_ROOT -iname "*.java"`
echo "Found java file $JAVAFILE"

# add newline to all the patch files
echo "Finding txt files in $PATCH_TAR_ROOT..."
find $PATCH_TAR_ROOT -iname "*.txt" -exec sed -i -e '$a\\' {} \;

# change newline character in java file (just in case)
echo "changing newline character in java file $JAVAFILE"
dos2unix $JAVAFILE

# apply patches
find $PATCH_TAR_ROOT -iname "*.txt" -exec sh -c "patch --dry-run $JAVAFILE {} --output={}.java" \;

# apply srcml
echo "parsing to AST..."
find $PATCH_TAR_ROOT -iname "*.java" -exec sh -c 'timeout 10s srcml --position {} > {}.xml' \; 

## Complete
echo "complete"
