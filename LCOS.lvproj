<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="11008008">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="Compute Spot Pattern v2.vi" Type="VI" URL="../Compute Spot Pattern v2.vi"/>
		<Item Name="Compute State Sequence.vi" Type="VI" URL="../Compute State Sequence.vi"/>
		<Item Name="Correct LCOS Pattern.vi" Type="VI" URL="../Correct LCOS Pattern.vi"/>
		<Item Name="Correction Pattern Changed.vi" Type="VI" URL="../Correction Pattern Changed.vi"/>
		<Item Name="Dispose LCOS Images.vi" Type="VI" URL="../Dispose LCOS Images.vi"/>
		<Item Name="LCOS Exit Global.vi" Type="VI" URL="../LCOS Exit Global.vi"/>
		<Item Name="LCOS GUI v3_green.vi" Type="VI" URL="../LCOS GUI v3_green.vi"/>
		<Item Name="LCOS GUI v3_red.vi" Type="VI" URL="../LCOS GUI v3_red.vi"/>
		<Item Name="LCOS Images Init.vi" Type="VI" URL="../LCOS Images Init.vi"/>
		<Item Name="LCOS Images.ctl" Type="VI" URL="../LCOS Images.ctl"/>
		<Item Name="LCOS Init.vi" Type="VI" URL="../LCOS Init.vi"/>
		<Item Name="LCOS Parent Action.ctl" Type="VI" URL="../LCOS Parent Action.ctl"/>
		<Item Name="LCOS Pattern Display Actions.ctl" Type="VI" URL="../LCOS Pattern Display Actions.ctl"/>
		<Item Name="LCOS Pattern Display_green.vi" Type="VI" URL="../LCOS Pattern Display_green.vi"/>
		<Item Name="LCOS Pattern Display_red.vi" Type="VI" URL="../LCOS Pattern Display_red.vi"/>
		<Item Name="LCOS Queue Management Actions.ctl" Type="VI" URL="../LCOS Queue Management Actions.ctl"/>
		<Item Name="LCOS Queue Manager.vi" Type="VI" URL="../LCOS Queue Manager.vi"/>
		<Item Name="LCOS State Machine States.ctl" Type="VI" URL="../LCOS State Machine States.ctl"/>
		<Item Name="Load LCOS Correction Pattern.vi" Type="VI" URL="../Load LCOS Correction Pattern.vi"/>
		<Item Name="Load LCOS Pattern.vi" Type="VI" URL="../Load LCOS Pattern.vi"/>
		<Item Name="Pattern_to_YAML.vi" Type="VI" URL="../Pattern_to_YAML.vi"/>
		<Item Name="Periodic Spot Pattern Parameters.ctl" Type="VI" URL="../Periodic Spot Pattern Parameters.ctl"/>
		<Item Name="python_pattern.vi" Type="VI" URL="../python_pattern.vi"/>
		<Item Name="python_pattern_script_node.vi" Type="VI" URL="../python_pattern_script_node.vi"/>
		<Item Name="Read Pattern from disk.vi" Type="VI" URL="../Read Pattern from disk.vi"/>
		<Item Name="Set LabPython Server Path.vi" Type="VI" URL="../Set LabPython Server Path.vi"/>
		<Item Name="test_python_pattern.vi" Type="VI" URL="../test_python_pattern.vi"/>
		<Item Name="Update LCOS Image.vi" Type="VI" URL="../Update LCOS Image.vi"/>
		<Item Name="Update Preview Image.vi" Type="VI" URL="../Update Preview Image.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="user.lib" Type="Folder">
				<Item Name="lvpython.dll" Type="Document" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/lvpython.dll"/>
				<Item Name="PYTHON Close Session__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Close Session__ogtk.vi"/>
				<Item Name="PYTHON Execute Script__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Execute Script__ogtk.vi"/>
				<Item Name="PYTHON Get Boolean Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Boolean Data__ogtk.vi"/>
				<Item Name="PYTHON Get Complex Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Complex Data__ogtk.vi"/>
				<Item Name="PYTHON Get Complex Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Complex Matrix__ogtk.vi"/>
				<Item Name="PYTHON Get Complex Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Complex Vector__ogtk.vi"/>
				<Item Name="PYTHON Get Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Data__ogtk.vi"/>
				<Item Name="PYTHON Get Float Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Float Data__ogtk.vi"/>
				<Item Name="PYTHON Get Float Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Float Matrix__ogtk.vi"/>
				<Item Name="PYTHON Get Float Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Float Vector__ogtk.vi"/>
				<Item Name="PYTHON Get Integer Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Integer Data__ogtk.vi"/>
				<Item Name="PYTHON Get Integer Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Integer Matrix__ogtk.vi"/>
				<Item Name="PYTHON Get Integer Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Integer Vector__ogtk.vi"/>
				<Item Name="PYTHON Get String Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get String Data__ogtk.vi"/>
				<Item Name="PYTHON Get String Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get String Vector__ogtk.vi"/>
				<Item Name="PYTHON Get Version and Types__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Get Version and Types__ogtk.vi"/>
				<Item Name="PYTHON New Session__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON New Session__ogtk.vi"/>
				<Item Name="PYTHON Session Refnum__ogtk.ctl" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Session Refnum__ogtk.ctl"/>
				<Item Name="PYTHON Set Boolean Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Boolean Data__ogtk.vi"/>
				<Item Name="PYTHON Set Complex Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Complex Data__ogtk.vi"/>
				<Item Name="PYTHON Set Complex Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Complex Matrix__ogtk.vi"/>
				<Item Name="PYTHON Set Complex Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Complex Vector__ogtk.vi"/>
				<Item Name="PYTHON Set Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Data__ogtk.vi"/>
				<Item Name="PYTHON Set Float Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Float Data__ogtk.vi"/>
				<Item Name="PYTHON Set Float Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Float Matrix__ogtk.vi"/>
				<Item Name="PYTHON Set Float Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Float Vector__ogtk.vi"/>
				<Item Name="PYTHON Set Integer Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Integer Data__ogtk.vi"/>
				<Item Name="PYTHON Set Integer Matrix__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Integer Matrix__ogtk.vi"/>
				<Item Name="PYTHON Set Integer Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Integer Vector__ogtk.vi"/>
				<Item Name="PYTHON Set Script Text__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Script Text__ogtk.vi"/>
				<Item Name="PYTHON Set Server Path__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set Server Path__ogtk.vi"/>
				<Item Name="PYTHON Set String Data__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set String Data__ogtk.vi"/>
				<Item Name="PYTHON Set String Vector__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON Set String Vector__ogtk.vi"/>
				<Item Name="PYTHON UTIL Format Error Code__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/labpython/labpython.llb/PYTHON UTIL Format Error Code__ogtk.vi"/>
				<Item Name="Tick Count (ms)__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/time/time.llb/Tick Count (ms)__ogtk.vi"/>
			</Item>
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Color (U64)" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Color (U64)"/>
				<Item Name="Color to RGB.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/colorconv.llb/Color to RGB.vi"/>
				<Item Name="ex_BuildTextVarProps.ctl" Type="VI" URL="/&lt;vilib&gt;/express/express output/BuildTextBlock.llb/ex_BuildTextVarProps.ctl"/>
				<Item Name="ex_CorrectErrorChain.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_CorrectErrorChain.vi"/>
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ ArrayToColorImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToColorImage"/>
				<Item Name="IMAQ ColorImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ColorImageToArray"/>
				<Item Name="IMAQ Copy" Type="VI" URL="/&lt;vilib&gt;/vision/Management.llb/IMAQ Copy"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Dispose" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Dispose"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Load Image Dialog" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Load Image Dialog"/>
				<Item Name="IMAQ ReadFile" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile"/>
				<Item Name="IMAQ SetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ SetImageSize"/>
				<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="SetMenuItemInfoSCConverter.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/SetMenuItemInfoSCConverter.vi"/>
			</Item>
			<Item Name="Init LCOS Menu.vi" Type="VI" URL="../Init LCOS Menu.vi"/>
			<Item Name="Launch DS Server if Local URL.vi" Type="VI" URL="../../../../../../Program Files (x86)/National Instruments/LabVIEW 2011/examples/comm/datasktcore.llb/Launch DS Server if Local URL.vi"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="Periodic Spot Pattern Parameters2.ctl" Type="VI" URL="../Periodic Spot Pattern Parameters2.ctl"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
