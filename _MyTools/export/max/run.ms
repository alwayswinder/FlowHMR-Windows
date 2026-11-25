-- This script resets scenes. Invoke with 3dsmaxbatch.exe only.
global rolBsRetarget
(
    local folder = getFilenamePath (getSourceFileName())
    local log = createFile (folder + "trial.txt")
    try (
        local motion = maxOps.mxsCmdLineArgs[#motion]
        if motion == undefined or not doesFileExist motion do throw "Missing input -mxsString motion"
        resetMaxFile #noPrompt
        frameRate = 30
        units.SystemType = #centimeters
        units.SystemScale = 1.0
        pluginManager.loadClass FBXIMP
        pluginManager.loadClass FBXEXP
        fileIn (folder + "BsRetargetTools.batch.ms")
        format "BsKeyTools loaded: %\n" rolBsRetarget to:log
        flush log
        local mapping = folder + "FlowHMR_Mapping.max"
        local skin = folder + "FlowHMR_Biped.max"
        local stage = maxOps.mxsCmdLineArgs[#stage]
        if stage != "export" do (
            FBXImporterSetParam "Mode" #create
            FBXImporterSetParam "Animation" false
            if not importFile (folder+"reference.fbx") #noPrompt using:FBXIMP do throw "Reference import failed"
            format "Reference imported, nodes=%\n" objects.count to:log
            rolBsRetarget.fnLoadMappingList (folder+"FlowHMR.list") quiet:true
            local state = rolBsRetarget.fnBuildMappingState()
            format "Mapping root=% required=% missingNodes=%\n" state.rootNode state.missingRequired state.missingSceneNodes to:log
            flush log
            if not state.canCreateMapping do throw "BsKeyTools rejected the mapping"
            rolBsRetarget.alignBipedToHelper.pressed()
            local root = rolBsRetarget.fnGetExistingBipedRoot()
            local helpers = for node in objects where matchPattern node.name pattern:"RTHelper_*" collect node
            if root == undefined or helpers.count < 30 do throw "BsKeyTools did not construct the mapping scene"
            format "Created % with % retarget helpers\n" root.name helpers.count to:log
            flush log
            if not saveMaxFile mapping quiet:true do throw "Mapping scene save failed"
            root.parent = undefined
            root.controller.figureMode = false
            local rig = #(root)
            local cursor = 1
            while cursor <= rig.count do (
                for child in rig[cursor].children do append rig child
                cursor += 1
            )
            format "Saving % target nodes, root class=%\n" rig.count (classOf root) to:log
            if rig.count < 30 or not saveNodes rig skin quiet:true do throw "Biped target scene save failed"
        )
        if not doesFileExist mapping or not doesFileExist skin do throw "Missing mapping or Biped scene"
        if stage == "export" do (
            if not loadMaxFile skin useFileUnits:true quiet:true do throw "Biped target scene load failed"
            local target = getNodeByName "Bip001" exact:true
            format "Target nodes=% Bip001=%\n" objects.count target to:log
            if target == undefined do throw "Expected Bip001 in the target scene"
            rolBsRetarget.edtMappingFile.text = mapping
            rolBsRetarget.edtSkinFile.text = skin
            rolBsRetarget.edtOutputPath.text = folder + "output"
            rolBsRetarget.ddlSaveVersion.selection = 1
            format "Starting BsKeyTools export\n" to:log
            flush log
            local result = rolBsRetarget.fnExportReTargrtedAnim #(motion) 1
            local bip = folder + "output/BIP/" + getFilenameFile motion + ".bip"
            format "BsKeyTools export=% BIP=% exists=%\n" result bip (doesFileExist bip) to:log
            if result != true or not doesFileExist bip do throw "BsKeyTools did not write a BIP"
        )
        format "FLOWHMR_OK\n" to:log
    ) catch (format "FAIL: %\n" (getCurrentException()) to:log)
    close log
)
