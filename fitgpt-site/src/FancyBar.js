import { colorCodes, colorMapping, lightColors } from './ColorData';

function FancyBar({ color, percent }) {
	const hexCode = colorCodes[color];
	const colorName = colorMapping[color];
	return (
		<div className="flex items-center justify-between">
			<div className="flex relative h-6 w-full rounded-full bg-gray-700 group">
				<div
					className="h-6 rounded-full"
					style={{
						width: `${percent}%`,
						backgroundColor: hexCode,
						border: color === "black" ? "1px solid gray" : ""
					}}
				/>
				<div
					className={`${percent >= 25 ? "absolute left-0 top-0 " : ""}h-6 text-white pl-3 text-right opacity-0 group-hover:opacity-100 transition-opacity`}
					style={{ color: lightColors.includes(color) && percent >= 25 ? "black" : "white"}}
				>
					{colorName}
				</div>
				
			</div>
			<p className="w-10 ml-2 text-right">{percent}%</p>
		</div>
	)
}

export default FancyBar;
