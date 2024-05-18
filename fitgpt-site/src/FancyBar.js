import { colorCodes, colorMapping, lightColors } from './ModelData';

function FancyBar({ color, percent }) {
	const hexCode = colorCodes[color];
	const colorName = colorMapping[color];
	return (
		<div className="flex items-center justify-between">
			<div className="progress flex items-center h-6 bg-gray-700 group border-2 border-gray-700">
				<div
					className="h-5 rounded-box"
					style={{
						width: `${percent}%`,
						backgroundColor: hexCode
					}}
				/>
				<p
					className={`${percent >= 25 ? "absolute left-0 " : ""}pl-3 text-left opacity-0 group-hover:opacity-100 transition-opacity`}
					style={{ color: lightColors.includes(color) && percent >= 25 ? "black" : "white"}}
				>
					{colorName}
				</p>
				
			</div>
			<p className="w-10 ml-2 text-right">{Math.round(percent)}%</p>
		</div>
	)
}

export default FancyBar;
