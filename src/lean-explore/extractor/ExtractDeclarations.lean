-- ExtractDeclarations.lean

import Lean
import Mathlib              -- Ensures Mathlib modules are loaded and accessible for analysis.
import Batteries.Lean.Json
import Batteries.Data.NameSet
import Lean.DocString           -- Provides `findDocString?`
import Lean.Environment         -- Provides `Environment` and related methods.
import Lean.Data.Lsp.Internal   -- Provides `DeclarationRanges` (via `References`).
import Lean.Data.Position       -- Provides `Position`.
import Lean.Util.Path           -- Provides `FilePath` operations.
import Lean.Util.Paths          -- Provides `initSrcSearchPath`.
import Lean.Server.References   -- Provides `findDeclarationRanges?`.
import Lean.Meta.Basic          -- Provides `MetaM`.
import Lean.Linter              -- Provides `isDeprecated`.
import Lean.Modifiers           -- Provides `isProtected`.
import Lean.ProjFns             -- Provides `Environment.isProjectionFn`.
import Std.Data.HashMap

open Lean Meta Std System IO FS Server References Linter -- Common Lean utilities.

/-!
# Lean Declaration, Location, and Dependency Extractor

This script processes an imported Lean environment, focusing on specified libraries
such as Mathlib, Batteries, Std, PhysLean, Init, and core Lean modules.
It extracts comprehensive information about declarations,
including their name, type, defining module, documentation string, source module,
source position (line and column numbers), status flags (e.g., protected, deprecated),
and attributes (specifically, the projection attribute). Additionally, it identifies
direct dependencies between these declarations.

The script is designed to avoid pretty-printing potentially large expressions (types or values)
and instead focuses on structural information and metadata. It includes progress reporting
for long-running operations and disables the Lean heartbeat limit to prevent timeouts.

The extracted data is output into two JSONL (JSON Lines) files:
- `declarations.jsonl`: Each line is a JSON object representing a single processed
  declaration and its associated metadata.
- `dependencies.jsonl`: Each line is a JSON object representing a direct dependency
  link between two processed declarations.
-/

/--
Recursively traverses a Lean expression (`e`) to find all direct constant dependencies.

Identifies `Expr.const` nodes, filtering out internal names and universe level constants
(e.g., `Type`, `Prop`, `Sort`). Results are collected into a `NameSet`.
-/
partial def findDirectDependencies (e : Expr) : MetaM NameSet := do
  let depRef ← IO.mkRef NameSet.empty
  forEachExpr' e fun subExpr => do
    if subExpr.isConst then
      let constName := subExpr.constName!
      -- Filter out internal names and universe level constants.
      if !(constName.isInternal ||
           constName == `Type || constName == `Prop || constName == `Sort) then
        depRef.modify fun currentDeps => currentDeps.insert constName
    return true
  depRef.get

/--
Processes the Lean environment to extract declaration and dependency information,
writing results to `hDecls` (declarations) and `hDeps` (dependencies).
Reports progress based on `totalToProcess`.
Accepts a `NameSet` of `targetRootNames` to filter declarations by module root.
-/
def extractData (hDecls hDeps : Handle) (totalToProcess : Nat) (targetRootNames : NameSet) : CoreM Unit := do
  let env ← getEnv
  let srcSearchPath ← initSrcSearchPath -- IO action lifted to CoreM.
  let allModules := env.allImportedModuleNames
  let processedCountRef ← IO.mkRef 0
  let progressInterval : Nat := 5000 -- Report progress every n declarations.

  /-
  Note: Pre-fetching of `simp` attribute state or instance checks was performed in earlier versions
  but has been removed.
  -/

  IO.println s!"Starting processing of {totalToProcess} declarations from specified modules..."

  env.constants.forM fun name constInfo => do
    let moduleIdx? : Option ModuleIdx := env.getModuleIdxFor? name
    let moduleName? : Option Name := moduleIdx?.bind fun idx => allModules[idx]?

    -- Skip internal declarations.
    if name.isInternal then return

    -- Process only declarations from the targetted modules.
    match moduleName? with
    | some modName =>
      unless targetRootNames.contains modName.getRoot do return
    | none => return -- Skip if no module name (e.g. some primitives not tied to a Lean file).

    processedCountRef.modify (· + 1)
    let currentCount ← processedCountRef.get

    if currentCount % progressInterval == 0 || currentCount == totalToProcess then
      IO.println s!"Processed {currentCount} / {totalToProcess} declarations..."

    let declTypeStr : String :=
      match constInfo with
      | .axiomInfo _   => "axiom"
      | .defnInfo _   => "definition"
      | .thmInfo _    => "theorem"
      | .opaqueInfo _ => "opaque"
      | .quotInfo _   => "quotient"
      | .inductInfo _ => "inductive"
      | .ctorInfo _   => "constructor"
      | .recInfo _    => "recursor"

    let moduleNameStr : Option String := moduleName?.map toString
    let docString? ← findDocString? env name

    -- Attempt to determine the relative file path of the source file.
    let sourcePath? : Option String ← match moduleName? with
      | some modName => do
          let pathOpt ← liftM <| Lean.SearchPath.findModuleWithExt srcSearchPath ".lean" modName -- IO action.
          match pathOpt with
          | some path =>
              -- Convert absolute path to relative to CWD.
              -- This basic replacement might need adjustment for complex project structures.
              let cwd ← IO.currentDir
              let relPath := path.toString.replace (cwd.toString ++ "/") ""
              pure (some relPath)
          | none => pure none
      | none => pure none

    -- Retrieve declaration's source range (start/end position) in `MetaM`.
    let rangePos? : Option (Position × Position) ← Meta.MetaM.run' do
        if let some ranges ← findDeclarationRanges? name then
          let declRange : DeclarationRange := ranges.range
          return some (declRange.pos, declRange.endPos)
        else return none

    let isProtected := Lean.isProtected env name
    let isDeprecated := Lean.Linter.isDeprecated env name

    -- Check specifically for the projection attribute.
    let attrsList : List String :=
      if Lean.Environment.isProjectionFn env name then -- Non-monadic check.
        ["projection"]
      else
        []

    let jsonAttrs := Json.arr <| attrsList.toArray.map Json.str

    let objFields : List (Option (String × Json)) := [
      some ("lean_name", Json.str name.toString),
      some ("decl_type", Json.str declTypeStr),
      moduleNameStr.map (fun s => ("module_name", Json.str s)),
      sourcePath?.map (fun s => ("source_file", Json.str s)),
      docString?.map (fun s => ("docstring", Json.str s)),
      rangePos?.map (fun (startPos, _) => ("range_start_line", Json.num startPos.line)),
      rangePos?.map (fun (startPos, _) => ("range_start_col", Json.num startPos.column)),
      rangePos?.map (fun (_, endPos) => ("range_end_line", Json.num endPos.line)),
      rangePos?.map (fun (_, endPos) => ("range_end_col", Json.num endPos.column)),
      some ("is_protected", Json.bool isProtected),
      some ("is_deprecated", Json.bool isDeprecated),
      some ("attributes", jsonAttrs)
    ]
    let jsonDecl := Json.mkObj <| objFields.filterMap id

    try
      hDecls.putStrLn jsonDecl.compress
    catch e =>
      let errorMsg ← e.toMessageData.toString
      IO.eprintln s!"Error writing declaration {name}: {errorMsg}"

    -- Extract direct dependencies from type and value (if applicable) in `MetaM`.
    let dependencyResult ← Meta.MetaM.run' do
        let typeDeps ← findDirectDependencies constInfo.type
        let valueDeps ← match constInfo.value? with
          | some value => findDirectDependencies value
          | none => pure NameSet.empty
        return typeDeps.union valueDeps

    for targetName in dependencyResult.toArray do
      -- Apply similar filters to dependencies: in target list, not self, not internal.
      let targetModuleIdx? : Option ModuleIdx := env.getModuleIdxFor? targetName
      let targetModuleName? : Option Name := targetModuleIdx?.bind fun idx => allModules[idx]?

      if targetName.isInternal then continue -- Skip internal dependencies.
      if targetName == name then continue -- Skip self-dependencies.

      match targetModuleName? with
      | some modName =>
        unless targetRootNames.contains modName.getRoot do continue -- Target not in our list.
      | none => continue -- Target has no module name, skip.

      let jsonDep := Json.mkObj [
        ("source_lean_name", Json.str name.toString),
        ("target_lean_name", Json.str targetName.toString),
        ("dependency_type", Json.str "Direct") -- All found are direct.
      ]
      try
        hDeps.putStrLn jsonDep.compress
      catch e =>
        let errorMsg ← e.toMessageData.toString
        IO.eprintln s!"Error writing dependency {name} -> {targetName}: {errorMsg}"


/--
Entry point for the extraction script.
Orchestrates setup, module import, pre-computation of declaration count,
and invokes `extractData`.
-/
unsafe def main : IO Unit := do
  let baseOutputDir : FilePath := ".." / "data"
  let declsFile : FilePath := baseOutputDir / "declarations.jsonl"
  let depsFile : FilePath := baseOutputDir / "dependencies.jsonl"

  IO.println s!"Starting location and dependency data extraction..."
  IO.println s!"Outputting declarations to: {declsFile}"
  IO.println s!"Outputting dependencies to: {depsFile}"

  -- Disable Lean heartbeat limit for long-running computations.
  let heartbeatLimit : Nat := 0
  IO.println s!"Heartbeat limit disabled (set to 0)."

  try
    IO.FS.createDirAll baseOutputDir
    -- Initialize Lean search path using toolchain's sysroot.
    Lean.initSearchPath (← Lean.findSysroot)

    let imports : Array Import := #[
      { module := `Mathlib },
      { module := `Batteries },
      { module := `FLT },
      { module := `PhysLean },
      { module := `Std }
    ]

    -- Import specified modules for analysis, configuring options like heartbeat limit.
    let opts : Options := {}
    let opts := maxHeartbeats.set opts heartbeatLimit
    IO.println s!"Importing specified modules (this may take a while)..."
    let env ← Lean.importModules imports (opts := opts) (trustLevel := 0)
    IO.println s!"Specified modules imported successfully. Environment created."

    let targetRootNames : NameSet := .ofList [
      `Mathlib, `Batteries, `PhysLean, `Std, `Init, `Lean, `FLT
    ]

    let targetRootNamesList : List Name := [`Mathlib, `Batteries, `PhysLean, `Std, `Init, `Lean, `FLT] -- Keep an ordered list for printing
    let initialCountsByRoot : Std.HashMap Name Nat := Id.run do
      let mut map := Std.HashMap.emptyWithCapacity targetRootNamesList.length
      for rootName in targetRootNamesList do
        map := map.insert rootName 0
      return map
    let countsByRootRef ← IO.mkRef initialCountsByRoot

    /- Pre-compute total number of declarations to process for accurate progress reporting. -/
    IO.println s!"Counting declarations to process from specified modules..."
    let allModules := env.allImportedModuleNames
    let totalToProcessRef ← IO.mkRef 0
    env.constants.forM fun name _ => do -- Only name needed for filters.
      if name.isInternal then return () -- Skip internal names.
      -- Apply same filtering as in `extractData` for accurate count.
      match env.getModuleIdxFor? name >>= fun idx => allModules[idx]? with
      | some modName =>
        let rootName := modName.getRoot
        if targetRootNames.contains rootName then
          totalToProcessRef.modify (· + 1)
          countsByRootRef.modify fun currentMap =>
          currentMap.insert rootName (currentMap.getD rootName 0 + 1)
      | none => return () -- Skip if no module name (e.g. some primitives).
    let totalToProcess ← totalToProcessRef.get
    IO.println s!"Found {totalToProcess} total declarations to process from specified modules."
    IO.println "Counts per module root:"
    let finalCountsByRoot ← countsByRootRef.get
    for rootName in targetRootNamesList do -- Iterate in the defined order for consistent output
      IO.println s!"- {rootName}: {finalCountsByRoot.getD rootName 0}"
    /- End of pre-computation. -/


    IO.FS.withFile declsFile IO.FS.Mode.write fun hDecls => do
      IO.FS.withFile depsFile IO.FS.Mode.write fun hDeps => do

        -- Setup `Core.Context` for `CoreM` actions, passing heartbeat limit.
        let ctx : Core.Context := { fileName := "<DeclarationInfoExtractor>", fileMap := default, maxHeartbeats := heartbeatLimit }

        -- Execute main data extraction logic in `CoreM`.
        match ← ((extractData hDecls hDeps totalToProcess targetRootNames).run ctx { env := env }).toIO' with
        | Except.ok _ =>
            IO.println s!"Processed {totalToProcess} / {totalToProcess} declarations..." -- Final confirmation.
            IO.println "Location and dependency data extraction completed successfully."
        | Except.error e =>
            throw <| IO.userError s!"CoreM execution failed: {← e.toMessageData.toString}"

  catch e =>
    IO.eprintln s!"Error during file handling or setup: {IO.Error.toString e}"
    IO.Process.exit 1 -- Exit with non-zero status on failure.
