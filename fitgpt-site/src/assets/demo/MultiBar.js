import { useState } from 'react';
import { colorCodes, colorMapping, articleMapping } from './ModelData';

function MultiBar({ article, data }) {
	const [hoveredSection, setHoveredSection] = useState(null);
	const [hoveredLabel, setHoveredLabel] = useState('');
	
	const sortedData = Object.entries(data)
		.map(([color, percent]) => ({ color, percent }))
		.sort((a, b) => b.percent - a.percent);
	
	// Function to handle mouse hover over a section
	const handleMouseEnter = (color, percent) => {
		setHoveredSection(color);
		setHoveredLabel(`${colorMapping[color]} (${(percent * 100).toFixed(2)}%)`);
	};

	return (
		<div className="space-y-1">
			<div className="flex justify-between items-center">
				<span className="font-semibold">{articleMapping[article]}</span>
				<span className={`transition-opacity ${hoveredSection ? 'opacity-100' : 'opacity-0'}`}>{hoveredLabel}</span>
			</div>
			<div className="progress flex h-6 border-2 border-gray-700 bg-gray-700">
				{sortedData.map(({ color, percent }) => (
					<div
						key={color}
						className="h-full"
						style={{
							width: `${percent * 100}%`,
							backgroundColor: colorCodes[color]
						}}
						onMouseEnter={() => handleMouseEnter(color, percent)}
						onMouseLeave={() => setHoveredSection(null)}
					/>
				))}
			</div>
		</div>
	);
}

export default MultiBar;
