<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="11008008">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="LCOS GUI v3.vi" Type="VI" URL="../LCOS GUI v3.vi"/>
		<Item Name="Compute Spot Pattern v2.vi" Type="VI" URL="../Compute Spot Pattern v2.vi"/>
		<Item Name="LCOS Pattern Display.vi" Type="VI" URL="../LCOS Pattern Display.vi"/>
		<Item Name="LCOS State Machine States.ctl" Type="VI" URL="../LCOS State Machine States.ctl"/>
		<Item Name="LCOS Init.vi" Type="VI" URL="../LCOS Init.vi"/>
		<Item Name="LCOS Pattern Display Actions.ctl" Type="VI" URL="../LCOS Pattern Display Actions.ctl"/>
		<Item Name="LCOS Queue Manager.vi" Type="VI" URL="../LCOS Queue Manager.vi"/>
		<Item Name="LCOS Queue Management Actions.ctl" Type="VI" URL="../LCOS Queue Management Actions.ctl"/>
		<Item Name="LCOS Exit Global.vi" Type="VI" URL="../LCOS Exit Global.vi"/>
		<Item Name="Load LCOS Correction Pattern.vi" Type="VI" URL="../Load LCOS Correction Pattern.vi"/>
		<Item Name="Load LCOS Pattern.vi" Type="VI" URL="../Load LCOS Pattern.vi"/>
		<Item Name="Periodic Spot Pattern Parameters.ctl" Type="VI" URL="../Periodic Spot Pattern Parameters.ctl"/>
		<Item Name="LCOS Images Init.vi" Type="VI" URL="../LCOS Images Init.vi"/>
		<Item Name="LCOS Images.ctl" Type="VI" URL="../LCOS Images.ctl"/>
		<Item Name="Dispose LCOS Images.vi" Type="VI" URL="../Dispose LCOS Images.vi"/>
		<Item Name="Correction Pattern Changed.vi" Type="VI" URL="../Correction Pattern Changed.vi"/>
		<Item Name="Correct LCOS Pattern.vi" Type="VI" URL="../Correct LCOS Pattern.vi"/>
		<Item Name="Compute State Sequence.vi" Type="VI" URL="../Compute State Sequence.vi"/>
		<Item Name="LCOS Parent Action.ctl" Type="VI" URL="../LCOS Parent Action.ctl"/>
		<Item Name="Update Preview Image.vi" Type="VI" URL="../Update Preview Image.vi"/>
		<Item Name="Update LCOS Image.vi" Type="VI" URL="../Update LCOS Image.vi"/>
		<Item Name="mulitspot_pattern_static3.vi" Type="VI" URL="../mulitspot_pattern_static3.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ SetImageSize" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ SetImageSize"/>
				<Item Name="Color (U64)" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Color (U64)"/>
				<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
				<Item Name="SetMenuItemInfoSCConverter.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/SetMenuItemInfoSCConverter.vi"/>
				<Item Name="IMAQ ReadFile" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile"/>
				<Item Name="IMAQ Load Image Dialog" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ Load Image Dialog"/>
				<Item Name="IMAQ Dispose" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Dispose"/>
				<Item Name="IMAQ ColorImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ColorImageToArray"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ Copy" Type="VI" URL="/&lt;vilib&gt;/vision/Management.llb/IMAQ Copy"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="Color to RGB.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/colorconv.llb/Color to RGB.vi"/>
				<Item Name="IMAQ ArrayToColorImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToColorImage"/>
			</Item>
			<Item Name="user.lib" Type="Folder">
				<Item Name="Tick Count (ms)__ogtk.vi" Type="VI" URL="/&lt;userlib&gt;/_OpenG.lib/time/time.llb/Tick Count (ms)__ogtk.vi"/>
			</Item>
			<Item Name="Init LCOS Menu.vi" Type="VI" URL="../Init LCOS Menu.vi"/>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="Periodic Spot Pattern Parameters2.ctl" Type="VI" URL="../Periodic Spot Pattern Parameters2.ctl"/>
			<Item Name="Launch DS Server if Local URL.vi" Type="VI" URL="../../../../../../Program Files (x86)/National Instruments/LabVIEW 2011/examples/comm/datasktcore.llb/Launch DS Server if Local URL.vi"/>
			<Item Name="pytscript.dll" Type="Document"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
