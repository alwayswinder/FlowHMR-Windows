-- Isolated Max Batch: correct knee planes, save BIP and sample the reopened file.
(
    local folder = getFilenamePath (getSourceFileName())
    local report = createFile (folder+"trial.txt")
    fn worldPos n = biped.getTransform n #pos
    fn worldRot n = biped.getTransform n #rotation
    fn kneeTurn thigh calf foot pose = (
        local hip = biped.getTransform thigh #pos
        local knee = biped.getTransform calf #pos
        local ankle = biped.getTransform foot #pos
        local axis = normalize (ankle-hip)
        local current = (knee-hip)-axis*(dot (knee-hip) axis)
        local sourceAxis = normalize (pose[3]-pose[1])
        local pole = (pose[2]-pose[1])-sourceAxis*(dot (pose[2]-pose[1]) sourceAxis)
        local desired = pole-axis*(dot pole axis)
        if length current < 0.01 or length desired < 0.01 then undefined else (
            current = normalize current
            desired = normalize desired
            atan2 (dot axis (cross current desired)) (dot current desired)
        )
    )
    fn requireNode name = (
        local n = getNodeByName name exact:true
        if n == undefined do throw ("Missing bone: "+name)
        n
    )
    fn checkRange rig frames = (
        local span = biped.getCurrentRange rig.controller
        if span.start.frame != 0 or span.end.frame != frames-1 do throw "Biped animation range differs from the source"
        span
    )
    try (
        local frames = maxOps.mxsCmdLineArgs[#frames] as integer
        local fps = maxOps.mxsCmdLineArgs[#fps] as integer
        resetMaxFile #noPrompt
        frameRate = fps
        units.SystemType = #centimeters
        units.SystemScale = 1.0
        pluginManager.loadClass FBXIMP
        FBXImporterSetParam "Mode" #create
        FBXImporterSetParam "Animation" true
        FBXImporterSetParam "FillTimeline" true
        if not importFile (folder+"motion.fbx") #noPrompt using:FBXIMP do throw "Could not read the FlowHMR motion"
        local srcLegs = for side in #("Left", "Right") collect (for suffix in #("UpLeg", "Leg", "Foot") collect requireNode (side+suffix))
        local source = for f=0 to frames-1 collect at time f (for leg in srcLegs collect (for n in leg collect n.transform.row4))
        if not loadMaxFile (folder+"output/MAX/motion.max") quiet:true useFileUnits:true do throw "Could not load retargeted Biped"
        local rig = requireNode "Bip001"
        local span = checkRange rig frames
        if frameRate != fps do throw "Biped frame rate differs from the source"
        local legs = for side in #(" L "," R ") collect (for suffix in #("Thigh","Calf","Foot","Toe0") collect requireNode (rig.name+side+suffix))
        local roots = for f=0 to frames-1 collect at time f (worldPos rig)
        local ankles = for f=0 to frames-1 collect at time f (for leg in legs collect #(worldPos leg[3],worldRot leg[3],worldRot leg[1]))
        local turns = for f=0 to frames-1 collect at time f (for s=1 to 2 collect kneeTurn legs[s][1] legs[s][2] legs[s][3] source[f+1][s])
        -- Keep a stable correction at straight frames where the knee plane is undefined.
        for s=1 to 2 do (
            local first = undefined
            for i=1 to turns.count while first==undefined where turns[i][s]!=undefined do first=turns[i][s]
            if first==undefined do first=0.0
            local previous=first
            for i=1 to turns.count do (
                local value=turns[i][s]
                if value==undefined then value=previous
                value += 360.0 * (floor ((previous-value)/360.0+0.5))
                turns[i][s]=value
                previous=value
            )
        )
        for f=0 to frames-1 do at time f (
            for s=1 to 2 do (
                local leg=legs[s]
                local target=ankles[f+1][s]
                local axis=normalize (target[1]-worldPos leg[1])
                local turn=0.25*turns[amax 1 f][s]+0.5*turns[f+1][s]+0.25*turns[amin frames (f+2)][s]
                local matrix=target[3] as matrix3
                local correction=(angleaxis (-turn) axis) as matrix3
                biped.setTransform leg[1] #rotation ((matrix*correction).rotationPart) true
                biped.setTransform leg[3] #rotation target[2] true
            )
        )
        local maxRootError=0.0
        local maxAnkleError=0.0
        for f=0 to frames-1 do at time f (
            maxRootError=amax maxRootError (distance (worldPos rig) roots[f+1])
            for s=1 to 2 do maxAnkleError=amax maxAnkleError (distance (worldPos legs[s][3]) ankles[f+1][s][1])
        )
        if maxRootError>0.1 or maxAnkleError>0.1 do throw "Knee correction displaced a foot or the root"
        local soles=for f=0 to frames-1 collect at time f (amin (for leg in legs collect (amin (nodeGetBoundingBox leg[3] (matrix3 1))[1].z (nodeGetBoundingBox leg[4] (matrix3 1))[1].z)))
        sort soles
        local lift=-soles[amax 1 ((soles.count*0.2) as integer)]
        with animate off (
            rig.controller.moveAllMode=true
            in coordsys world move rig [0,0,lift]
            biped.collapseMoveAllMode rig.controller
            rig.controller.moveAllMode=false
        )
        local expectedRoot=at time 0f (worldPos rig)
        local path=folder+"biped.bip"
        biped.saveBipFile rig.controller path
        if not doesFileExist path do throw "Could not save BIP"
        if not loadMaxFile (folder+"FlowHMR_Biped.max") quiet:true useFileUnits:true do throw "Could not open the Biped reference for readback"
        frameRate=fps
        rig=requireNode "Bip001"
        biped.loadBipFile rig.controller path #matchFileStruct #noRedraw
        local readbackSpan=checkRange rig frames
        local readbackRoot=at time 0f (worldPos rig)
        if distance readbackRoot expectedRoot>0.1 do throw ("BIP root mismatch: expected " + expectedRoot as string + ", read " + readbackRoot as string)
        animationRange=readbackSpan
        sliderTime=readbackSpan.start
        if not saveMaxFile (folder+"biped.max") quiet:true do throw "Could not save the animated Biped scene"
        local nodes=#(rig)
        local parents=#(-1)
        local j=1
        while j<=nodes.count do (
            for child in nodes[j].children where classOf child==Biped_Object and not matchPattern child.name pattern:"* Footsteps" do (
                append nodes child
                append parents (j-1)
            )
            j+=1
        )
        if nodes.count<45 do throw "Incomplete Biped readback"
        local metadata=createFile (folder+"skeleton.tsv")
        for i=1 to nodes.count do format "%\t%\n" nodes[i].name parents[i] to:metadata
        close metadata
        local positions=createFile (folder+"samples.tsv")
        for tick=0 to ((frames-1)*ticksPerFrame) by (ticksPerFrame/4) do at time (1t*tick) (
            for node in nodes do (
                local p=worldPos node
                local tip=p+[0,2.8,0]*(worldRot node as matrix3)
                format "% % % % % % " p.x p.y p.z tip.x tip.y tip.z to:positions
            )
            format "\n" to:positions
        )
        close positions
        format "FLOWHMR_OK\n" to:report
    ) catch (format "FAIL: %\n" (getCurrentException()) to:report)
    close report
)
