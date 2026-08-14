-- ExtractAST.lean

import Lean
import Lake
import Lean.Data.Json.FromToJson -- For parsing JSON configuration.
import Std.Data.HashMap          -- For counting files per library

/-!
# Lean File Tactic and Premise Data Extractor

This script is a modified version of data extraction tools from the LeanDojo
project:
https://github.com/lean-dojo/LeanDojo/blob/v2.2.0/src/lean_dojo/data_extraction/ExtractData.lean

It processes Lean source
files from specified libraries (e.g., Mathlib, Batteries, Std, Lean Core, etc.)
to extract detailed trace information, primarily focusing on:
- Abstract Syntax Trees (ASTs) for each command, augmented with pre-calculated
  full byte spans.
- Tactic applications, including their source positions and the tactic
  state (goals) before and after execution.
- Premise usage, identifying definitions and theorems used, along with their
  definition locations and usage sites within the analyzed file.

Key features include:
- Configuration-driven targeting of specific Lean libraries.
- Robust path resolution for project files, Lake dependencies, and Lean core libraries.
- In-depth analysis via traversal of Lean's `InfoTree`.
- Parallel processing for efficient extraction across multiple files in a project.
- Organized output into a common `data/AST/` directory structure.

For each processed `.lean` file (e.g., `MyModule.lean` from library `MyLib`),
the script generates:
1.  **`.ast.json` file**: A JSON object (e.g., `data/AST/MyLib/MyModule.ast.json`)
    containing the collected ASTs (with spans), tactic traces, and premise traces,
    mirroring the source file's relative path within its library.
2.  **`.dep_paths` file**: A text file (e.g., `data/AST/MyLib/MyModule.dep_paths`)
    listing the source file paths of all modules imported by the processed file,
    similarly placed in the output directory.

The script is intended to be run using Lake:
`lake env lean --run ExtractAST.lean [processProject]`
(If no argument is given, it defaults to `processProject`.)
The command `processSingleFileTask` is used internally for parallel execution.
A configuration file `extractor_config.json` is used to specify the
Lean toolchain source path.
-/

open Lean Elab System IO FS Lean.FromJson

set_option maxHeartbeats 2000000  -- 10x the default maxHeartbeats.


instance : ToJson Substring where
  toJson s := toJson s.toString

instance : ToJson String.Pos where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax
deriving instance ToJson for Position

namespace LeanExplore

/--
Configuration loaded from `extractor_config.json`.
Specifies the root directory of the Lean toolchain's source files.
-/
structure ExtractorConfig where
  toolchainCoreSrcDir : String
  deriving FromJson

/--
Represents a Lean library to be processed by the extractor.
-/
structure LibraryToProcess where
  name : String              /- User-friendly name for the library, used in output paths, e.g., "Mathlib", "Init". -/
  srcRootDir : FilePath      /- Absolute path to the root of its .lean source files. -/
  deriving Repr

/--
Represents the trace of a single tactic application.
-/
structure TacticTrace where
  stateBefore: String        /- The tactic state before the tactic was applied. -/
  stateAfter: String         /- The tactic state after the tactic was applied. -/
  pos: String.Pos            /- The start position of the tactic's syntax in the source file. -/
  endPos: String.Pos         /- The end position of the tactic's syntax in the source file. -/
  deriving ToJson


/--
Represents the trace of a premise (a definition or theorem used).
-/
structure PremiseTrace where
  fullName: String           /- The fully-qualified name of the premise. -/
  defPos: Option Position    /- The start position of the premise's definition, if available. -/
  defEndPos: Option Position /- The end position of the premise's definition, if available. -/
  modName: String            /- The name of the module where the premise is defined. -/
  defPath: String            /- The file path where the premise is defined. -/
  pos: Option Position       /- The start position of the premise's usage in the current file, if available. -/
  endPos: Option Position    /- The end position of the premise's usage in the current file, if available. -/
  deriving ToJson

/--
Encapsulates a Lean command's `Syntax` object along with its pre-calculated
full byte span in the source file.
-/
structure CommandSyntaxWithSpan where
  commandSyntax : Syntax         /- The raw `Syntax` object for the command. -/
  byteStart : Option String.Pos  /- The UTF-8 byte offset of the start of the command's syntax. `none` if not determinable. -/
  byteEnd : Option String.Pos    /- The UTF-8 byte offset of the end of the command's syntax. `none` if not determinable. -/
  deriving ToJson

/--
Aggregates all tracing information for a Lean file.
-/
structure Trace where
  commandASTs : Array CommandSyntaxWithSpan /- An array of command ASTs, each with its pre-calculated byte span. -/
  tactics: Array TacticTrace             /- An array of all tactic traces recorded from the file. -/
  premises: Array PremiseTrace           /- An array of all premise traces recorded from the file. -/
  deriving ToJson


/--
The monad used for accumulating trace information.
It is a state monad (`StateT Trace`) layered over `MetaM`.
-/
abbrev TraceM := StateT Trace MetaM


namespace Pp

/--
Appends a newline character to a string if the string is not empty.
-/
private def addLine (s : String) : String :=
  if s.isEmpty then s else s ++ "\n"


/-
Pretty-prints a goal to a string.
This version is similar to `Meta.ppGoal` but uses `String` instead of `Format`
to ensure that local declarations are consistently separated by newline characters.
-/
private def ppGoal (mvarId : MVarId) : MetaM String := do
  match (← getMCtx).findDecl? mvarId with
  | none          => return "unknown goal"
  | some mvarDecl =>
    let indent          := 2
    let lctx            := mvarDecl.lctx
    let lctx            := lctx.sanitizeNames.run' { options := (← getOptions) }
    Meta.withLCtx lctx mvarDecl.localInstances do
      let rec pushPending (ids : List Name) (type? : Option Expr) (s : String) : MetaM String := do
        if ids.isEmpty then
          return s
        else
          let s := addLine s
          match type? with
          | none      => return s
          | some type =>
            let typeFmt ← Meta.ppExpr type
            return (s ++ (Format.joinSep ids.reverse (format " ") ++ " :" ++ Format.nest indent (Format.line ++ typeFmt)).group).pretty
      let rec ppVars (varNames : List Name) (prevType? : Option Expr) (s : String) (localDecl : LocalDecl) : MetaM (List Name × Option Expr × String) := do
        match localDecl with
        | .cdecl _ _ varName type _ _ =>
          let varName := varName.simpMacroScopes
          let type ← instantiateMVars type
          if prevType? == none || prevType? == some type then
            return (varName :: varNames, some type, s)
          else do
            let s ← pushPending varNames prevType? s
            return ([varName], some type, s)
        | .ldecl _ _ varName type val _ _ => do
          let varName := varName.simpMacroScopes
          let s ← pushPending varNames prevType? s
          let s  := addLine s
          let type ← instantiateMVars type
          let typeFmt ← Meta.ppExpr type
          let mut fmtElem  := format varName ++ " : " ++ typeFmt
          let val ← instantiateMVars val
          let valFmt ← Meta.ppExpr val
          fmtElem := fmtElem ++ " :=" ++ Format.nest indent (Format.line ++ valFmt)
          let s := s ++ fmtElem.group.pretty
          return ([], none, s)
      let (varNames, type?, s) ← lctx.foldlM (init := ([], none, "")) fun (varNames, prevType?, s) (localDecl : LocalDecl) =>
          if localDecl.isAuxDecl || localDecl.isImplementationDetail then
            return (varNames, prevType?, s)
          else
            ppVars varNames prevType? s localDecl
      let s ← pushPending varNames type? s
      let goalTypeFmt ← Meta.ppExpr (← instantiateMVars mvarDecl.type)
      let goalFmt := Meta.getGoalPrefix mvarDecl ++ Format.nest indent goalTypeFmt
      let s := s ++ "\n" ++ goalFmt.pretty
      match mvarDecl.userName with
      | Name.anonymous => return s
      | name           => return "case " ++ name.eraseMacroScopes.toString ++ "\n" ++ s

/--
Pretty-prints a list of goals, separated by double newlines.
Returns "no goals" if the list is empty.
-/
def ppGoals (ctx : ContextInfo) (goals : List MVarId) : IO String :=
  if goals.isEmpty then
    return "no goals"
  else
    let fmt := ctx.runMetaM {} (return Std.Format.prefixJoin "\n\n" (← goals.mapM (ppGoal ·)))
    return (← fmt).pretty.trim


end Pp


namespace Path

/--
Computes `path` relative to `parent`.
Returns `none` if `path` is not hierarchically under `parent`,
or if one path is absolute and the other is relative.
-/
def relativeTo (path parent : FilePath) : Option FilePath :=
  let rec componentsRelativeTo (pathComps parentComps : List String) : Option FilePath :=
    match pathComps, parentComps with
    | _, [] => mkFilePath pathComps
    | [], _ => none
    | (h₁ :: t₁), (h₂ :: t₂) =>
      if h₁ == h₂ then
        componentsRelativeTo t₁ t₂
      else
        none
  if path.isAbsolute != parent.isAbsolute then
    none -- Cannot determine relative path if one is absolute and the other is not.
  else
    componentsRelativeTo path.components parent.components


/--
Checks if `path` is relative to `parent`.
Returns true if `relativeTo path parent` returns `some _`, false otherwise.
-/
def isRelativeTo (path parent : FilePath) : Bool :=
  match relativeTo path parent with
  | some _ => true
  | none => false


/--
Converts a `FilePath` to an absolute path.
If the path is already absolute, it is returned unchanged.
Otherwise, it is resolved against the current working directory.
-/
def toAbsolute (path : FilePath) : IO FilePath := do
  if path.isAbsolute then
    pure path
  else
    let cwd ← IO.currentDir
    pure $ cwd / path


/--
Normalizes a file path by removing "." components.
This function is useful for both relative and absolute paths.
-/
private def trim (path : FilePath) : FilePath :=
  mkFilePath $ path.components.filter (· != ".")


/--
The directory where Lake stores downloaded packages.
Typically `.lake/packages` or a user-configured `packagesDir`.
-/
def packagesDir : FilePath :=
  if Lake.defaultPackagesDir == "packages"  then
    ".lake" / Lake.defaultPackagesDir
  else
    Lake.defaultPackagesDir


/--
The default build directory used by Lake.
Its specific location (e.g., `.lake/build` or `build`) can vary based on the Lean version.
-/
def buildDir : FilePath :=
  if Lake.defaultPackagesDir.fileName == "packages" then  -- Corresponds to Lean >= v4.3.0-rc2 behavior
    ".lake/build"
  else  -- Corresponds to Lean < v4.3.0-rc2 behavior
    "build"

/--
The directory within the Lake build directory where compiled Lean libraries (.olean, .ilean, etc.) are typically stored.
-/
def libDir : FilePath := buildDir / "lib" / "lean"


/--
Converts a build artifact path (e.g., an `.olean` file path) back to its corresponding source file path
with the specified extension (`ext`).

This function handles:
1. Files from the Lean toolchain (resolved via `Lean.findSysroot`).
2. Files from Lean 4 as a local source dependency (e.g. `.lake/packages/lean4/...`).
3. Files from other packaged dependencies in `.lake/packages/<pkg>/...`.
4. Project-local files (e.g., `project_root/.lake/build/lib/lean/MyModule.olean`).

Returns an `IO FilePath` due to potential file system operations like `Lean.findSysroot`.
Throws an `IO.userError` if the source path cannot be determined.
-/
def toSrcDir! (oleanPath : FilePath) (ext : String) : IO FilePath := do
  let normalizedOleanPath := oleanPath.normalize -- Use normalized paths for consistent comparisons.

  -- 1. Handle files belonging to the Lean toolchain.
  let sysroot ← Lean.findSysroot
  let toolchainLibDir := (sysroot / "lib" / "lean").normalize
  -- Assuming toolchain sources are under sysroot/src/lean/ (for Init, Std, Lean modules as per user's findings)
  let toolchainSrcParentDir := (sysroot / "src" / "lean").normalize

  let toolchainLibDirStr := toolchainLibDir.toString
  let normalizedOleanPathStr := normalizedOleanPath.toString

  if normalizedOleanPathStr.startsWith toolchainLibDirStr then
    let relativePathStr := normalizedOleanPathStr.drop toolchainLibDirStr.length
    let sep := System.FilePath.pathSeparator.toString
    let relStr :=
      if relativePathStr.startsWith sep then
        relativePathStr.drop sep.length
      else
        relativePathStr
    let relativePath := FilePath.mk relStr
    -- The source path needs to be relative to toolchainSrcParentDir
    -- e.g. if olean is .../lib/lean/Init/Core.olean, source is .../src/lean/Init/Core.lean
    let srcPath := toolchainSrcParentDir / relativePath
    return System.FilePath.withExtension srcPath ext

  let trimmedOleanPath := trim oleanPath

  -- 2. Handle cases where Lean 4 itself is a local source dependency (less common).
  let lean4PackageLibDir := packagesDir / "lean4" / "lib"
  if let some p := relativeTo trimmedOleanPath lean4PackageLibDir then
    return System.FilePath.withExtension (LeanExplore.Path.packagesDir / "lean4" / "src" / p) ext

  -- 3. Handle files from other packages located in `.lake/packages`.
  if trimmedOleanPath.toString.startsWith packagesDir.toString then
    let pathRelToPackagesRoot := mkFilePath (trimmedOleanPath.components.drop packagesDir.components.length)
    if pathRelToPackagesRoot.components.isEmpty then
      throw $ IO.userError s!"toSrcDir!: Path relative to packagesDir is empty for {oleanPath}"
    let pkgName := pathRelToPackagesRoot.components.head!
    let pkgBuildPathPrefixes : List FilePath := [
      FilePath.mk pkgName / ".lake" / "build" / "lib" / "lean",
      FilePath.mk pkgName / ".lake" / "build" / "lib"
    ]
    for pfx in pkgBuildPathPrefixes do
      if let some p := relativeTo trimmedOleanPath (packagesDir / pfx) then
        return (packagesDir / pkgName / p).withExtension ext
    if pathRelToPackagesRoot.components.head? == some pkgName then
        return (packagesDir / pathRelToPackagesRoot).withExtension ext
    throw $ IO.userError s!"toSrcDir!: Could not determine source path for package file {oleanPath} within package {pkgName}"

  -- 4. Handle files belonging to the current project.
  let projectBuildLibDir := libDir
  if let some p := relativeTo trimmedOleanPath projectBuildLibDir then
    return System.FilePath.withExtension p ext

  throw $ IO.userError s!"toSrcDir!: Could not determine source for olean path '{oleanPath}'. CWD: {← IO.currentDir}."


/--
Ensures all parent directories of the given file path `p` exist, creating them if necessary.
-/
def makeParentDirs (p : FilePath) : IO Unit := do
  let some parent := p.parent | pure () -- No parent means root or single component, no dir to create.
  IO.FS.createDirAll parent


/--
Locates the `.lean` source file corresponding to a given module name.
It handles standard modules, as well as special `«lake-packages»` and `«.lake»` prefixed modules.
Relies on `findOLean` and `Path.toSrcDir!` for the conversion.
Throws an `IO.userError` if the source file cannot be found or does not exist.
-/
def findLean (mod : Name) : IO FilePath := do
  let modStr := mod.toString
  if modStr.startsWith "«lake-packages»." then
    let relativePathStr := modStr.drop "«lake-packages».".length |>.replace "." FilePath.pathSeparator.toString
    let p := packagesDir / relativePathStr |>.withExtension "lean"
    if !(← p.pathExists) then
        throw $ IO.userError s!"Path.findLean: Constructed path {p} for special module {mod} does not exist."
    return p
  if modStr.startsWith "«.lake»." then
    let relativePathStr := modStr.drop "«.lake».".length |>.replace "." FilePath.pathSeparator.toString
    let p := (FilePath.mk ".lake") / relativePathStr |>.withExtension "lean"
      if !(← p.pathExists) then
        throw $ IO.userError s!"Path.findLean: Constructed path {p} for special module {mod} does not exist."
    return p

  let oleanPath ← findOLean mod
  let leanPath ← Path.toSrcDir! oleanPath "lean"

  if !(← leanPath.pathExists) then
    let cwd ← IO.currentDir
    throw $ IO.userError s!"Path.findLean: Derived source path '{leanPath}' for module '{mod}' (from .olean at '{oleanPath}') does not exist. CWD: {cwd}."
  return leanPath

end Path


namespace Traversal

/--
Processes `TacticInfo` nodes within the `InfoTree` to extract tactic execution traces.
It focuses on tactics that cause a change in the goal state.
-/
private def visitTacticInfo (ctx : ContextInfo) (ti : TacticInfo) (parent : InfoTree) : TraceM Unit := do
  match ti.stx.getKind with
  | ``Lean.Parser.Term.byTactic =>
    match ti.stx with
    | .node _ _ #[.atom _ "by", .node _ ``Lean.Parser.Tactic.tacticSeq _] => pure ()
    | _ => assert! false -- Should not happen with `by` tactics
  | ``Lean.Parser.Tactic.tacticSeq =>
    match ti.stx with
    | .node _ _ #[.node _ ``Lean.Parser.Tactic.tacticSeq1Indented _] => pure ()
    | .node _ _ #[.node _ ``Lean.Parser.Tactic.tacticSeqBracketed _] => pure ()
    | _ => assert! false -- Should not happen with tactic sequences
  | _ => pure ()

  match parent with
  | .node (Info.ofTacticInfo i) _ =>
    match i.stx.getKind with
    | ``Lean.Parser.Tactic.tacticSeq1Indented | ``Lean.Parser.Tactic.tacticSeqBracketed | ``Lean.Parser.Tactic.rewriteSeq =>
      let ctxBefore := { ctx with mctx := ti.mctxBefore }
      let ctxAfter := { ctx with mctx := ti.mctxAfter }
      let stateBefore ← Pp.ppGoals ctxBefore ti.goalsBefore
      let stateAfter ← Pp.ppGoals ctxAfter ti.goalsAfter
      if stateBefore == "no goals" || stateBefore == stateAfter then
        pure () -- No change in goal state or no goals to begin with.
      else
        let some posBefore := ti.stx.getPos? true | pure ()
        let some posAfter := ti.stx.getTailPos? true | pure ()
        match ti.stx with
        | .node _ _ _ =>
          modify fun trace => {
            trace with tactics := trace.tactics.push {
              stateBefore := stateBefore,
              stateAfter := stateAfter,
              pos := posBefore,
              endPos := posAfter,
              }
          }
        | _ => pure ()
    | _ => pure ()
  | _ => pure ()


/--
Processes `TermInfo` nodes from the `InfoTree` to identify and trace premises
(constants or definitions referenced in the code).
-/
private def visitTermInfo (ti : TermInfo) (env : Environment) : TraceM Unit := do
  let some fullName := ti.expr.constName? | return () -- Only interested in constants/definitions.
  let fileMap ← getFileMap

  let posBefore := match ti.toElabInfo.stx.getPos? with
    | some posInfo => fileMap.toPosition posInfo
    | none => none
  let posAfter := match ti.toElabInfo.stx.getTailPos? with
    | some posInfo => fileMap.toPosition posInfo
    | none => none

  let decRanges ← withEnv env $ findDeclarationRanges? fullName
  let defPos := decRanges.map (fun decR => decR.selectionRange.pos)
  let defEndPos := decRanges.map (fun decR => decR.selectionRange.endPos)

  let modName :=
  if let some modIdx := env.const2ModIdx.get? fullName then
    env.header.moduleNames[modIdx.toNat]!
  else
    env.header.mainModule

  let defPath : String ← liftM (m:=IO) do toString <$> LeanExplore.Path.findLean modName

  if defPos != posBefore ∧ defEndPos != posAfter then
    modify fun trace => {
        trace with premises := trace.premises.push {
          fullName := toString fullName,
          defPos := defPos,
          defEndPos := defEndPos,
          defPath := defPath,
          modName := toString modName,
          pos := posBefore,
          endPos := posAfter,
        }
    }

/--
Dispatches to specific visitor functions (`visitTacticInfo`, `visitTermInfo`)
based on the type of `Info` encountered in the `InfoTree`.
-/
private def visitInfo (ctx : ContextInfo) (i : Info) (parent : InfoTree) (env : Environment) : TraceM Unit := do
  match i with
  | .ofTacticInfo ti => visitTacticInfo ctx ti parent
  | .ofTermInfo ti => visitTermInfo ti env
  | _ => pure () -- Other info types are not processed for tracing.


/--
Recursively traverses an `InfoTree`, processing nodes using `visitInfo`
and accumulating trace data in the `TraceM` state.
It correctly handles nested contexts within the tree.
-/
private partial def traverseTree (ctx: ContextInfo) (tree : InfoTree)
(parent : InfoTree) (env : Environment) : TraceM Unit := do
  match tree with
  | .context ctx' t =>
    match ctx'.mergeIntoOuter? ctx with
    | some ctx' => traverseTree ctx' t tree env
    | none => panic! "Failed to synthesize contextInfo when traversing infoTree"
  | .node i children =>
    visitInfo ctx i parent env
    for x in children do
      traverseTree ctx x tree env
  | _ => pure () -- Leaf nodes or other tree structures not explicitly handled.

/--
Initiates the traversal for a top-level `InfoTree` node,
establishing the initial context.
-/
private def traverseTopLevelTree (tree : InfoTree) (env : Environment) : TraceM Unit := do
  match tree with
  | .context ctx t =>
    match ctx.mergeIntoOuter? none with
    | some ctx => traverseTree ctx t tree env
    | none => panic! "Failed to synthesize contextInfo for top-level infoTree"
  | _ => pure () -- Top-level node is expected to be a context node or handled appropriately.


/--
Traverses an array of `InfoTree`s (an "info forest"), typically representing
all information collected for a file. It aggregates all trace data into a single `Trace` object.
-/
def traverseForest (trees : Array InfoTree) (env : Environment) : TraceM Trace := do
  for t in trees do
    traverseTopLevelTree t env
  get

end Traversal


open Traversal

/--
Extracts and formats import statements from the given module header `Syntax`.
It resolves module names to their source file paths using `findOLean` and `Path.toSrcDir!`.
The paths are returned as a newline-separated string.
-/
def getImports (header: Syntax) : IO String := do
  let mut s := ""
  for dep in headerToImports header do
    let oleanPath ← Lean.findOLean dep.module
    let leanPath ← Path.toSrcDir! oleanPath "lean"
    if !(← leanPath.pathExists) then
      IO.eprintln s!"Warning: getImports: Derived source path {leanPath} for import {dep.module} (from .olean {oleanPath}) does not exist."
      continue

    if leanPath.isRelative then
      s := s ++ "\n" ++ leanPath.toString
    else if ¬(oleanPath.toString.endsWith "/lib/lean/Init.olean") && ¬(oleanPath.toString.endsWith "\\lib\\lean\\Init.olean") then -- Avoids Init from core.
      s := s ++ "\n" ++ leanPath.toString
  return s.trim


/--
Core logic to process a single Lean source file and extract trace data.
Writes the trace to `jsonOutputPath` and dependency paths to `depOutputPath`.
Accepts the pre-calculated `mainModuleName`.
-/
unsafe def extractDataForFile (inputPath : FilePath) (jsonOutputPath : FilePath) (depOutputPath : FilePath) (mainModuleName : Name) : IO Unit := do
  println! s!"Processing: {inputPath} (module: {mainModuleName}) -> {jsonOutputPath}, {depOutputPath}"
  let input ← IO.FS.readFile inputPath
  enableInitializersExecution
  let inputCtx := Parser.mkInputContext input inputPath.toString
  let (headerSyntax, parserState, messages) ← Parser.parseHeader inputCtx -- Renamed `header` to `headerSyntax` for clarity
  let (env, messages) ← processHeader headerSyntax {} messages inputCtx

  if messages.hasErrors then
    for msg in messages.toList do
      if msg.severity == .error then
        IO.eprintln s!"ERROR in {inputPath} (module: {mainModuleName}): {← msg.toString}"
    throw $ IO.userError "Errors during import processing; aborting file trace."

  let env := env.setMainModule mainModuleName
  let commandState := { Command.mkState env messages {} with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState commandState
  let env' := s.commandState.env
  let originalCommandsArray := s.commands.pop -- This is Array Syntax, contains commands after header
  let infoTreesArray := s.commandState.infoState.trees.toArray

  -- Prepare CommandSyntaxWithSpan array
  let mut commandASTsWithSpan : Array CommandSyntaxWithSpan := #[]

  -- Process header syntax
  let headerSpanOpt := match headerSyntax.getPos?, headerSyntax.getTailPos? with
    | some start, some tail => some (start, tail)
    | _, _ => none
  if headerSpanOpt.isNone then
    IO.eprintln s!"Warning: Could not retrieve full span for header syntax in module {mainModuleName} (file: {inputPath}). Header: {headerSyntax.reprint.getD "<no reprint available>"}"
  commandASTsWithSpan := commandASTsWithSpan.push {
    commandSyntax := headerSyntax,
    byteStart := headerSpanOpt.map Prod.fst,
    byteEnd := headerSpanOpt.map Prod.snd
  }

  -- Process main commands syntax
  for cmdStx in originalCommandsArray do
    let cmdSpanOpt := match cmdStx.getPos?, cmdStx.getTailPos? with
      | some start, some tail => some (start, tail)
      | _, _ => none
    if cmdSpanOpt.isNone then
      IO.eprintln s!"Warning: Could not retrieve full span for a command's syntax in module {mainModuleName} (file: {inputPath}). Command Syntax: {cmdStx.reprint.getD "<no reprint available>"}"
    commandASTsWithSpan := commandASTsWithSpan.push {
      commandSyntax := cmdStx,
      byteStart := cmdSpanOpt.map Prod.fst,
      byteEnd := cmdSpanOpt.map Prod.snd
    }

  -- Initialize the Trace state for traversal using the new `commandASTsWithSpan`
  let traceMInitState : Trace := { commandASTs := commandASTsWithSpan, tactics := #[], premises := #[] }
  let traceM := (traverseForest infoTreesArray env').run' traceMInitState -- Pass the fully initialized state

  let (finalTraceState, _) ← traceM.run'.toIO {fileName := inputPath.toString, fileMap := FileMap.ofString input} {env := env}

  Path.makeParentDirs jsonOutputPath
  IO.FS.writeFile jsonOutputPath (toJson finalTraceState).pretty

  Path.makeParentDirs depOutputPath
  IO.FS.writeFile depOutputPath (← getImports headerSyntax) -- Use original headerSyntax for imports

/--
Loads the extractor configuration from `extractor_config.json`.
Falls back to default (sysroot) if config file or key is missing.
-/
def loadExtractorConfig : IO ExtractorConfig := do
  let configPath : FilePath := "extractor_config.json"
  if !(← configPath.pathExists) then
    IO.eprintln s!"Warning: Configuration file '{configPath}' not found. Using default Lean toolchain source path."
    let sysroot ← Lean.findSysroot
    return { toolchainCoreSrcDir := (sysroot / "src").toString }

  let configContent ← FS.readFile configPath
  match Json.parse configContent >>= fromJson? (α := ExtractorConfig) with
  | .ok config => return config
  | .error err =>
    IO.eprintln s!"Warning: Could not parse '{configPath}': {err}. Using default Lean toolchain source path."
    let sysroot ← Lean.findSysroot
    return { toolchainCoreSrcDir := (sysroot / "src").toString }

/--
Generates the list of libraries to process based on the extractor configuration.
-/
def getTargetLibraries (config : ExtractorConfig) : IO (Array LibraryToProcess) := do
  let toolchainSrcPath := FilePath.mk config.toolchainCoreSrcDir
  let mut libs : Array LibraryToProcess := #[]

  -- Toolchain libraries
  libs := libs.push { name := "Init", srcRootDir      := ← Path.toAbsolute (toolchainSrcPath / "lean" / "Init") }
  libs := libs.push { name := "Std", srcRootDir       := ← Path.toAbsolute (toolchainSrcPath / "lean" / "Std") }
  libs := libs.push { name := "Lean", srcRootDir      := ← Path.toAbsolute (toolchainSrcPath / "lean" / "Lean") }

  -- Lake packages
  let packagesDirPath ← Path.toAbsolute Path.packagesDir
  libs := libs.push { name := "Batteries", srcRootDir := packagesDirPath / "batteries" }
  libs := libs.push { name := "Mathlib", srcRootDir   := packagesDirPath / "mathlib" }
  libs := libs.push { name := "PhysLean", srcRootDir  := packagesDirPath / "PhysLean" }

  -- Verify existence of srcRootDir for all configured libraries
  let mut validLibs : Array LibraryToProcess := #[]
  for lib in libs do
    if ← lib.srcRootDir.isDir then
      validLibs := validLibs.push lib
    else
      IO.eprintln s!"Warning: Source directory for library '{lib.name}' not found at '{lib.srcRootDir}'. Skipping."
  return validLibs

/--
Calculates the target output path for a processed file.
The path will be relative to `baseDojoDir`, structured as `<baseDojoDir>/<libName>/<relativeSourcePath>.<targetExtension>`.
-/
def getTargetOutputPath
  (baseDojoDir : FilePath)      --(absolute path)
  (libName : String)
  (libSrcRootDir : FilePath)    --(absolute path)
  (absSourceFilePath : FilePath) --(absolute path)
  (targetFileExtension : String)
  : IO FilePath := do
  let normalizedLibSrcRootDir := libSrcRootDir.normalize
  let normalizedAbsSourceFilePath := absSourceFilePath.normalize

  match Path.relativeTo normalizedAbsSourceFilePath normalizedLibSrcRootDir with
  | none =>
    throw <| IO.userError s!"Source file '{normalizedAbsSourceFilePath}' is not relative to library root '{normalizedLibSrcRootDir}'."
  | some relativeSrcPath =>
    let relativeOutputPathWithNewExt := relativeSrcPath.withExtension targetFileExtension
    let finalOutputPath := baseDojoDir / libName / relativeOutputPathWithNewExt
    return finalOutputPath.normalize


/--
Formats a duration given in milliseconds into a human-readable string (e.g., "1h 23m 45s").
-/
def formatDurationMs (ms : Nat) : String :=
  let secs_total := ms / 1000
  let mins_total := secs_total / 60
  let hours_total := mins_total / 60

  let s := secs_total % 60
  let m := mins_total % 60
  let h := hours_total

  if h > 0 then
    s!"{h}h {m}m {s}s"
  else if m > 0 then
    s!"{m}m {s}s"
  else
    s!"{s}s"


/--
Orchestrates the processing of all configured Lean libraries.
It discovers `.lean` files within each library, checks their eligibility,
calculates output paths, and manages a pool of worker processes for parallel extraction.
It will skip processing a file if its corresponding `.ast.json` output already exists.
-/
def processProject : IO Unit := do
  let sysroot ← Lean.findSysroot
  _ ← Lean.initSearchPath sysroot
  let config ← loadExtractorConfig
  let targetLibraries ← getTargetLibraries config
  let cwd ← Process.getCurrentDir
  let baseDojoDir ← Path.toAbsolute (cwd / ".." / "data" / "AST")
  IO.FS.createDirAll baseDojoDir
  IO.println s!"Outputting extracted data to: {baseDojoDir}"

  let mut filesToProcessParams : Array (FilePath × FilePath × FilePath × String × String) := #[]
  let mut eligibleFilesCountByLibrary : Std.HashMap String Nat := ∅
  IO.println "[INFO] Identifying all eligible files and checking for existing outputs..."

  for lib in targetLibraries do
    if !(← lib.srcRootDir.isDir) then
      IO.eprintln s!"Warning: Source directory for library '{lib.name}' does not exist or is not a directory: {lib.srcRootDir}'. Skipping."
      continue

    for leanFilePathAbs in ← System.FilePath.walkDir lib.srcRootDir do
      if leanFilePathAbs.extension != some "lean" then
        continue

      let relativeModuleName? ← try
        some <$> moduleNameOfFileName leanFilePathAbs (some lib.srcRootDir)
      catch e =>
        IO.eprintln s!"Error determining relative module name for '{leanFilePathAbs}' (lib: {lib.name}, root: {lib.srcRootDir}): {toString e}. Skipping."
        pure none

      if relativeModuleName?.isNone then continue
      let relativeModuleName := relativeModuleName?.get!

      let fullModuleName : Name :=
        match lib.name with
        | "Init"   => (Name.mkSimple "Init").append relativeModuleName
        | "Std"    => (Name.mkSimple "Std").append relativeModuleName
        | "Lean"   => (Name.mkSimple "Lean").append relativeModuleName
        | _        => relativeModuleName

      let oleanPath? ← try
        let _ ← Lean.findOLean fullModuleName
        pure (some ())
      catch _ =>
        pure none

      if oleanPath?.isSome then
        -- This file is eligible (has an .olean), increment count for the library.
        let currentTotalCountForLib := eligibleFilesCountByLibrary.getD lib.name 0
        eligibleFilesCountByLibrary := eligibleFilesCountByLibrary.insert lib.name (currentTotalCountForLib + 1)

        -- Determine the path for the .ast.json output file.
        let astJsonPath ← getTargetOutputPath baseDojoDir lib.name lib.srcRootDir leanFilePathAbs "ast.json"

        -- Check if the .ast.json file already exists.
        if ← astJsonPath.pathExists then
          pure ()
          -- IO.println s!"[INFO] Skipping '{leanFilePathAbs}' (module: {fullModuleName}): Output '{astJsonPath}' already exists."
        else
          -- .ast.json does not exist, so add this file to the processing queue.
          let depPathsPath ← getTargetOutputPath baseDojoDir lib.name lib.srcRootDir leanFilePathAbs "dep_paths"
          let taskDescr := s!"{lib.name} / {leanFilePathAbs.fileName.getD "file"} (module: {fullModuleName.toString.takeRight 40})"
          filesToProcessParams := filesToProcessParams.push (leanFilePathAbs, astJsonPath, depPathsPath, fullModuleName.toString, taskDescr)
        -- No 'else' needed for oleanPath? check; if no .olean, it's skipped entirely.
      -- End of oleanPath?.isSome check
    -- End of loop through leanFilePathAbs
  -- End of loop through targetLibraries

  IO.println "\n[INFO] Eligible files per library (files with corresponding .olean):"
  for libSpec in targetLibraries do
    IO.println s!"  - {libSpec.name}: {eligibleFilesCountByLibrary.getD libSpec.name 0}"
  IO.println ""

  if filesToProcessParams.isEmpty then
    IO.println "[INFO] No files found to process (either none eligible or all outputs already exist)."
    return

  let maxWorkers : Nat := 32
  let mut nextFileIndex : Nat := 0
  let totalFilesToProcess := filesToProcessParams.size -- This now reflects files that NEED processing.

  let mut activeTasksData : Array (Task (Except IO.Error Process.Output) × FilePath × FilePath × String) := #[]

  let mut overallProcessedCount := 0
  let mut overallFailedTasksCount := 0

  IO.println s!"[INFO] Starting processing of {totalFilesToProcess} files (outputs not found) using up to {maxWorkers} concurrent workers."
  let startTimeMs ← IO.monoMsNow -- Record start time for ETR

  while nextFileIndex < totalFilesToProcess || !activeTasksData.isEmpty do
    let mut reapedTasksThisIteration := false
    let mut launchedTasksThisIteration := false

    let mut stillRunningTasks := #[]
    for (task, astJsonPath, depPathsPath, taskDescr) in activeTasksData do
      if ← IO.hasFinished task then
        reapedTasksThisIteration := true
        let previousFilesHandled := overallProcessedCount + overallFailedTasksCount

        match ← IO.wait task with
        | Except.error err =>
          overallFailedTasksCount := overallFailedTasksCount + 1
          IO.eprintln s!"ERROR (task execution) for '{taskDescr}': {err}"
        | Except.ok processOutput =>
          if processOutput.exitCode != 0 then
            overallFailedTasksCount := overallFailedTasksCount + 1
            IO.eprintln s!"ERROR (worker process) for '{taskDescr}' exited with code {processOutput.exitCode}."
            unless processOutput.stdout.isEmpty do IO.eprintln s!"    Stdout:\n---\n{processOutput.stdout}\n---"
            unless processOutput.stderr.isEmpty do IO.eprintln s!"    Stderr:\n---\n{processOutput.stderr}\n---"
          else
            overallProcessedCount := overallProcessedCount + 1
            if !(← astJsonPath.pathExists) then IO.eprintln s!"  [VERIFY ERROR] AST JSON file NOT found for '{taskDescr}': {astJsonPath}"
            if !(← depPathsPath.pathExists) then IO.eprintln s!"  [VERIFY ERROR] Dep Paths file NOT found for '{taskDescr}': {depPathsPath}"

        let filesHandledSoFar := overallProcessedCount + overallFailedTasksCount
        let printIntervalMessage := (filesHandledSoFar > 0 && (filesHandledSoFar % 10 == 0)) || (filesHandledSoFar == totalFilesToProcess && filesHandledSoFar > previousFilesHandled)

        if printIntervalMessage then
            let currentTimeMs ← IO.monoMsNow
            let timePassedMs := currentTimeMs - startTimeMs
            let timePassedStr := formatDurationMs timePassedMs

            let mut logLine := s!"Processed {filesHandledSoFar}/{totalFilesToProcess} files. (Time passed: {timePassedStr}"
            if filesHandledSoFar > 0 && filesHandledSoFar < totalFilesToProcess then
                let avgTimePerFileMs : Float := timePassedMs.toFloat / filesHandledSoFar.toFloat
                let filesRemaining := totalFilesToProcess - filesHandledSoFar
                let etrMs : Float := avgTimePerFileMs * filesRemaining.toFloat
                if etrMs >= 0 then
                    let etrStr := formatDurationMs (etrMs.toUInt64).toNat
                    logLine := logLine ++ s!", ETR: {etrStr}"
            logLine := logLine ++ ")"
            IO.println logLine
      else
        stillRunningTasks := stillRunningTasks.push (task, astJsonPath, depPathsPath, taskDescr)
    activeTasksData := stillRunningTasks

    while activeTasksData.size < maxWorkers && nextFileIndex < totalFilesToProcess do
      launchedTasksThisIteration := true
      let (leanFilePathAbs, astJsonPath, depPathsPath, fullModuleNameStr, taskDescr) := filesToProcessParams[nextFileIndex]!
      let spawnArgs : Process.SpawnArgs := {
        cmd := "lake",
        args := #["env", "lean", "--run", "ExtractAST.lean", "processSingleFileTask", leanFilePathAbs.toString, astJsonPath.toString, depPathsPath.toString, fullModuleNameStr],
        stdout := Process.Stdio.piped, stderr := Process.Stdio.piped
      }
      let newTask ← IO.asTask (Process.output spawnArgs) (Task.Priority.dedicated)
      activeTasksData := activeTasksData.push (newTask, astJsonPath, depPathsPath, taskDescr)
      nextFileIndex := nextFileIndex + 1

    if !activeTasksData.isEmpty then
      if !reapedTasksThisIteration && !launchedTasksThisIteration then
        let currentLeanTasks : List (Task (Except IO.Error Process.Output)) := activeTasksData.toList.map (·.1)
        if hNotEmpty : currentLeanTasks.length > 0 then
          let _ ← IO.waitAny currentLeanTasks hNotEmpty
        else
          IO.eprintln "[WARN] Active tasks reported, but task list for waitAny was empty. Sleeping briefly."
          IO.sleep (UInt32.ofNat 100)
      else if activeTasksData.size == maxWorkers && nextFileIndex < totalFilesToProcess && (reapedTasksThisIteration || launchedTasksThisIteration) then
        IO.sleep (UInt32.ofNat 10)
    else if nextFileIndex == totalFilesToProcess && activeTasksData.isEmpty then
      let filesHandledSoFar := overallProcessedCount + overallFailedTasksCount
      if filesHandledSoFar > 0 && (filesHandledSoFar % 10 != 0 || filesHandledSoFar != totalFilesToProcess) then
          let currentTimeMs ← IO.monoMsNow
          let timePassedMs := currentTimeMs - startTimeMs
          let timePassedStr := formatDurationMs timePassedMs
          IO.println s!"Processed {filesHandledSoFar}/{totalFilesToProcess} files. (Time passed: {timePassedStr})"

      IO.println "[INFO] All files processed and all tasks completed."
      break
    else if !launchedTasksThisIteration && !reapedTasksThisIteration && nextFileIndex < totalFilesToProcess && activeTasksData.isEmpty then
      IO.sleep (UInt32.ofNat 20)

  let finalTimeMs ← IO.monoMsNow
  let totalDurationMs := finalTimeMs - startTimeMs
  IO.println s!"\n[INFO] Processing summary:"
  IO.println s!"Total files targeted for processing (i.e., needing output): {totalFilesToProcess}."
  IO.println s!"Successfully processed files: {overallProcessedCount}."
  IO.println s!"Failed tasks/processes: {overallFailedTasksCount}."
  IO.println s!"Total processing time: {formatDurationMs totalDurationMs}."

  if overallFailedTasksCount > 0 then
    IO.eprintln s!"{overallFailedTasksCount} tasks/processes reported failure. Review error messages above."
    Process.exit 1
  else if overallProcessedCount == totalFilesToProcess then
    IO.println "All targeted files processed successfully."
    IO.println s!"Please check the directory '{baseDojoDir}' for output files."
  else
    let unhandledOrUnaccounted := totalFilesToProcess - overallProcessedCount - overallFailedTasksCount
    IO.println s!"Processing completed. Note: {unhandledOrUnaccounted} files may not have been fully processed or accounted for."
    IO.println s!"Please check the directory '{baseDojoDir}' for output files."


/--
Parses a dot-separated string into a Lean Name.
e.g., "Foo.Bar.Baz" becomes `Foo.Bar.Baz
-/
def stringToName (s : String) : Name :=
  if s == "[anonymous]" then Name.anonymous
  else s.splitOn "." |>.foldl (fun acc part => Name.mkStr acc part) Name.anonymous

/--
Worker function called by `processProject` via CLI to process a single file.
-/
unsafe def runProcessSingleFileTask (inputPathStr astJsonPathStr depPathsPathStr fullModuleNameStr : String) : IO Unit := do
  let inputPath := FilePath.mk inputPathStr
  let astJsonPath := FilePath.mk astJsonPathStr
  let depPathsPath := FilePath.mk depPathsPathStr
  let mainModuleName := stringToName fullModuleNameStr
  extractDataForFile inputPath astJsonPath depPathsPath mainModuleName

end LeanExplore


open LeanExplore

/--
Main entry point for the LeanExplore data extraction script.
-/
unsafe def main (args : List String) : IO Unit := do
  match args with
  | ["processProject"] => processProject
  | "processSingleFileTask" :: inputPathStr :: astJsonPathStr :: depPathsPathStr :: fullModuleNameStr :: [] =>
    runProcessSingleFileTask inputPathStr astJsonPathStr depPathsPathStr fullModuleNameStr
  | [] => processProject -- Default to processing the project
  | _ =>
    throw $ IO.userError s!"Invalid arguments: {args}. Usage: \n  lean --run ExtractAST.lean [processProject]\nOR (internally for worker processes):\n  lean --run ExtractAST.lean processSingleFileTask <input_path> <ast_json_output_path> <dep_paths_output_path> <full_module_name_string>"