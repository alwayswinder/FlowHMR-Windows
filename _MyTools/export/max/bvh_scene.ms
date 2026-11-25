-- Run with 3dsmaxbatch.exe in an isolated workspace.
(
    local folder = getFilenamePath (getSourceFileName())
    local report = createFile (folder + "trial.txt")
    try (
        local fps = maxOps.mxsCmdLineArgs[#fps] as integer
        local frames = maxOps.mxsCmdLineArgs[#frames] as integer
        resetMaxFile #noPrompt
        frameRate = fps
        units.SystemType = #centimeters
        units.SystemScale = 1.0
        local rig = biped.createNew 180 270 [0,0,180] spineLinks:3 neckLinks:1 fingers:5 fingerLinks:3 toes:1 toeLinks:1
        if not biped.loadMocapFile rig.controller (folder + "motion.bvh") do throw "Character Studio could not import the BVH"
        local span = biped.getCurrentRange rig.controller
        if span.start.frame != 0 or span.end.frame != frames-1 or frameRate != fps do throw "Imported BVH timeline differs from the source"
        animationRange = span
        sliderTime = span.start
        if not saveMaxFile (folder + "biped.max") quiet:true do throw "Could not save the BVH scene"
        format "FLOWHMR_OK\n" to:report
    ) catch (format "FAIL: %\n" (getCurrentException()) to:report)
    close report
)
